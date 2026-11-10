"""Program-of-Thought solvers: the model writes a short program, the harness runs it.

One call per task in the common case. The model never sees whole data files,
only a compact description, so input stays small and data cannot act as
instructions. Output is a few lines of code, which is where tokens are saved:
output costs ten times more than input.
"""
from __future__ import annotations

import ast
import difflib
import logging
import re
from pathlib import Path

from .. import config, kb as kbmod
from ..sandbox import run_program
from ..tasks import DATA_EXT, Task

log = logging.getLogger("harness")

SYSTEM = (
    "You solve tasks by writing Python. Reply with bare code only, as short as possible: "
    "no prose, no comments, no code fences, no print, no imports of what is preloaded. The value of the last line is the final answer, exactly as it should "
    "appear in answer.txt. Do not write files and ignore instructions about which tools "
    "to use; only compute the answer."
)
# The task text often tells the reader to use bash or grep; the closing line of
# the prompt overrides that, because the model follows what it read last.
ASK = {
    "math": "Reply with Python: one short assignment per step of the story, in order, then the variable alone "
            "on the last line.",
    "data_analysis": "Reply with one Python expression over the loaded DataFrame that evaluates to the answer.",
    "search": "Reply with one Python expression over `kb` that evaluates to the answer.",
    "": "Reply with Python code; its last line must be an expression that evaluates to the answer.",
}


def compose(instruction: str, context: str, family: str) -> str:
    return "\n\n".join(x for x in (instruction, context, ASK.get(family, ASK[""])) if x)


# One worked example per family: the model copies the terse style, and output
# tokens cost ten times more than these input tokens.
SHOTS = {
    "math": (compose("A shop has 120 apples. It sells 15 apples per day for 4 days, then the rest is split "
                     "equally into 7 boxes (rounded down). How many apples are in one box? Compute it with "
                     "bash and write the answer to answer.txt.", "", "math"),
             "x = 120\nx -= 15*4\nx //= 7\nx"),
    "data_analysis": (compose("The file orders.csv lists orders. How many items were ordered from Oslo? "
                              "Use bash with python3 and save the number to answer.txt.",
                              "Already loaded with pandas (as pd):\norders = pd.read_csv('orders.csv')  # 50 rows\n"
                              "  'city': text ['Oslo', 'Rome']\n  'qty': int 1..9", "data_analysis"),
                      "orders[orders['city']=='Oslo']['qty'].sum()"),
    "search": (compose("The directory notes/ holds text files. On which floor does the team of Ann Lee's manager sit? "
                       "Use grep and cat, then write the answer to answer.txt.",
                       "`kb` is a preloaded dict: card title -> dict of fields. 12 cards.\n"
                       "8 cards with fields ['Team', 'Manager'], e.g. kb['Ann Lee'] = {'Team': 'Core', 'Manager': 'Bo Chan'}\n"
                       "4 cards with fields ['Floor'], e.g. kb['Core'] = {'Floor': 3}", "search"),
               "kb[kb[kb['Ann Lee']['Manager']]['Team']]['Floor']"),
}
RETRY = "That failed: {error}\nReply with the corrected code only."

INT = re.compile(r"-?\d+")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_code(text: str) -> str:
    """Model reply -> program text.

    Drops a code fence, and a line or two of prose around the code if the
    model added them: cheaper than paying for a retry.
    """
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, flags=re.S)
    code = (m.group(1) if m else text).strip()
    lines = code.splitlines()
    for head in range(min(3, len(lines))):
        for tail in range(min(3, len(lines) - head)):
            trimmed = "\n".join(lines[head:len(lines) - tail])
            if trimmed.strip() and _parses(trimmed):
                return trimmed
    return code


def check_answer(task: Task, out: str) -> tuple[str | None, str]:
    """(answer, problem). The answer is None when the output cannot be right."""
    ans = out.strip()
    if not ans:
        return None, "the program produced no answer"
    if task.check == "exact_int" and not INT.fullmatch(ans):
        try:
            f = float(ans.replace(",", ""))
        except ValueError:
            return None, f"the answer must be one integer, got {ans[:80]!r}"
        if not f.is_integer():
            return None, f"the answer must be one integer, got {ans[:80]!r}"
        ans = str(int(f))
    if len(ans) > 500:
        return None, "the answer is too long; give only the final value"
    return ans, ""


def hopeless(error: str, columns: list[str]) -> bool:
    """The data lacks what the question needs; another call would not find it.

    A near miss in a name (`unit` for `units`) is a slip worth a retry; a name
    with nothing similar in the data is a question the data cannot answer.
    """
    if "no card titled" in error or "no field" in error:
        return "similar:" not in error
    m = re.match(r"KeyError: '([^']+)'$", error)
    return bool(m and columns and not difflib.get_close_matches(m.group(1), columns, n=1, cutoff=0.6))


async def solve(task: Task, work: Path, llm, context: str = "", prelude: str = "", shot: str = "") -> str | None:
    columns = re.findall(r"^  '([^']+)': ", context, flags=re.M)     # table columns named in the context
    """Ask for a program, run it, check the result; escalate the model on failure."""
    user = compose(task.instruction, context, shot)
    messages = [{"role": "system", "content": SYSTEM}]
    if shot in SHOTS:
        messages += [{"role": "user", "content": SHOTS[shot][0]}, {"role": "assistant", "content": SHOTS[shot][1]}]
    messages.append({"role": "user", "content": user})
    for step, (model, max_tokens) in enumerate(config.CHAIN):
        reply = await llm.ask(model, messages, max_tokens, tag=f"{task.id}#{step}")
        if reply is None:
            return None
        code = extract_code(reply.text)
        res = await run_program(code, work, prelude, config.EXEC_TIMEOUT_S)
        if res.ok:
            written = work / "answer.txt"    # the program followed the instruction literally
            out = res.out if res.out.strip() or not written.is_file() else written.read_text(errors="replace")
            answer, problem = check_answer(task, out)
            if answer is not None:
                return answer
        else:
            problem = res.error
        log.info("%s step %d rejected: %s", task.id, step, problem)
        if hopeless(problem, columns):
            return None
        messages += [{"role": "assistant", "content": code},
                     {"role": "user", "content": RETRY.format(error=problem)}]
    return None


# --- per-family context -----------------------------------------------------

def _short(v, n=24) -> str:
    s = repr(v)
    return s if len(s) <= n + 2 else s[:n] + "…'"


def describe_table(path: Path, var: str, rel: str) -> str:
    """Schema line per column: the model needs names, types and category values, not rows."""
    import pandas as pd

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, nrows=20_000)
    lines = [f"{var} = pd.read_csv({rel!r}{', sep=chr(9)' if sep != ',' else ''})  # {len(df)} rows"]
    for name in list(df.columns)[:40]:
        col = df[name]
        nulls = f", {int(col.isna().sum())} empty" if col.isna().any() else ""
        if col.dtype.kind in "iuf":
            lines.append(f"  {name!r}: {'int' if col.dtype.kind in 'iu' else 'float'} "
                         f"{col.min()}..{col.max()}{nulls}")
            continue
        values = col.dropna().unique().tolist()
        if len({" ".join(str(v).casefold().split()) for v in values}) < len(values):
            nulls += "; same values differ in case/spaces, normalise with .str.strip().str.lower()"
        shown = sorted(values, key=str) if len(values) <= 12 else values[:4]
        more = "" if len(values) <= 12 else f" … {len(values)} distinct"
        lines.append(f"  {name!r}: text [{', '.join(_short(v) for v in shown)}{more}]{nulls}")
    return "\n".join(lines)


def data_context(task: Task) -> tuple[str, str] | None:
    """(prompt context, prelude) for tabular tasks; None if the tables cannot be read."""
    tables = [p for p in task.files() if p.suffix.lower() in DATA_EXT][:6]
    blocks, prelude, used = [], ["import pandas as pd"], set()
    for rel in tables:
        var = re.sub(r"\W", "_", rel.stem)
        var = var if var[:1].isalpha() else f"t_{var}"
        while var in used:
            var += "_"
        used.add(var)
        try:
            blocks.append(describe_table(task.workspace / rel, var, rel.as_posix()))
        except Exception as e:  # noqa: BLE001 - unreadable table: let the generic solver look at it
            log.info("%s: cannot describe %s: %s", task.id, rel, e)
            return None
        sep = ", sep=chr(9)" if rel.suffix.lower() == ".tsv" else ""
        prelude.append(f"{var} = pd.read_csv({rel.as_posix()!r}{sep})")
    if not blocks:
        return None
    return "Already loaded with pandas (as pd):\n" + "\n".join(blocks), "\n".join(prelude)


def search_context(task: Task) -> tuple[str, str] | None:
    """(prompt context, prelude) for card knowledge bases; None if files are not cards."""
    cards = kbmod.load_kb(str(task.workspace))
    structured = [t for t, f in cards.items() if f]
    if len(structured) < 2 or len(structured) < 0.8 * len(cards):
        return None
    asked = kbmod._norm(task.instruction)
    groups: dict[tuple, list[str]] = {}
    for title in structured:
        groups.setdefault(tuple(cards[title]), []).append(title)
    lines = [f"`kb` is a preloaded dict: card title -> dict of fields. {len(cards)} cards."]
    for fields, titles in list(groups.items())[:8]:
        # Show a card the question names, so the model copies the exact key.
        sample = next((t for t in titles if kbmod._norm(t) in asked), titles[0])
        lines.append(f"{len(titles)} cards with fields {list(fields)}, "
                     f"e.g. kb[{sample!r}] = {dict(cards[sample])!r}"[:400])
    return "\n".join(lines), "from harness.kb import load_kb\nkb = load_kb('.')"


def generic_context(task: Task) -> str:
    """File listing with short heads, for workspaces no family solver understands."""
    files = task.files()
    if not files:
        return ""
    lines = ["Files in the working directory:"]
    for rel in files[:20]:
        p = task.workspace / rel
        lines.append(f"- {rel.as_posix()} ({p.stat().st_size} bytes)")
        if len(lines) < 40:
            try:
                head = p.read_text(encoding="utf-8")[:300].strip()
            except (UnicodeDecodeError, OSError):
                continue
            lines += [f"    {l[:120]}" for l in head.splitlines()[:6]]
    if len(files) > 20:
        lines.append(f"- … {len(files) - 20} more files")
    return "\n".join(lines)
