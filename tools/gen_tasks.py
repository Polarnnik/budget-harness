"""Generate a training task set in the organisers' format.

    python tools/gen_tasks.py --out runs/train --per-type 15 --seed 1

Four families as in the examples, with what the final set is expected to add:
other wordings, other question kinds, dirty data, unsolvable tasks and traps.
Every task.json carries `expected` and `solvable`; coding tasks also get
hidden_tests/, wider than the visible tests. The harness reads none of these.
"""
import argparse
import csv
import json
import random
from pathlib import Path

TAILS = [
    "Work only inside the current directory. When you are done, write ONLY the final answer to the file "
    "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).",
    "Stay inside the working directory. Save just the result, nothing else, into answer.txt.",
    "Put the final value (and only it) in a file named answer.txt in this folder.",
]


def emit(out: Path, kind: str, i: int, instruction: str, expected, check: str, rng, solvable=True) -> Path:
    d = out / f"{kind}_{i:04d}"
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({
        "task_id": d.name, "task_type": kind, "instruction": f"{instruction} {rng.choice(TAILS)}",
        "reward": rng.randint(1, 5), "expected": str(expected), "check": check, "solvable": solvable,
    }, indent=2))
    return d


# --- math ---------------------------------------------------------------------

def gen_math(out, i, rng):
    x = rng.randint(800, 5000)
    unit = rng.choice(["boxes", "crates", "pallets"])
    text = [rng.choice([f"A warehouse starts with {x} {unit}.", f"Initially a depot holds {x} {unit}.",
                        f"There are {x} {unit} in storage at the start."])]
    broken = rng.random() < 0.15
    for step in range(rng.randint(2, 5)):
        op = rng.choice(["ship", "receive", "split", "double", "damage", "percent"])
        if op == "ship":
            k, d = rng.randint(5, 40), rng.randint(2, 6)
            x -= k * d
            if broken and step == 0:
                text.append(f"It ships some {unit} every day for {d} days.")
                continue
            text.append(rng.choice([f"It ships {k} {unit} per day for {d} days.",
                                    f"Over the next {d} days, {k} {unit} leave every day."]))
        elif op == "receive":
            n, m = rng.randint(2, 6), rng.randint(20, 120)
            x += n * m
            text.append(rng.choice([f"It receives {n} deliveries of {m} {unit} each.",
                                    f"{n} trucks arrive, each bringing {m} {unit}."]))
        elif op == "split":
            n = rng.randint(2, 4)
            x //= n
            text.append(rng.choice([
                f"Then the stock is split equally into {n} warehouses and it keeps one share (rounded down).",
                f"Next the {unit} are divided evenly among {n} sites and only one site's share stays; drop any remainder."]))
        elif op == "double":
            x *= 2
            text.append(rng.choice(["Then the stock is doubled by merging with a sister warehouse.",
                                    "After that the stock doubles: a sister depot with exactly as many as it has now merges in."]))
        elif op == "damage":
            k = rng.randint(3, 60)
            x -= k
            text.append(f"{k} {unit} are found damaged and thrown away.")
        else:
            p = rng.choice([10, 20, 25, 50])
            x -= x * p // 100
            text.append(f"A flood destroys {p}% of the stock (round the loss down).")
    text.append(rng.choice([f"How many {unit} does it have at the end?", f"What is the final number of {unit}?"]))
    emit(out, "math", i, " ".join(text), x, "exact_int", rng, solvable=not broken)


# --- data analysis ------------------------------------------------------------

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]


def gen_data(out, i, rng):
    n = rng.choice([20, 60, 300])
    dirty = rng.random() < 0.2
    rows = [{"order_id": k + 1, "region": rng.choice(REGIONS), "product": rng.choice(PRODUCTS),
             "month": rng.randint(1, 12), "units": rng.randint(1, 40), "price": rng.randint(5, 99)}
            for k in range(n)]
    p, r = rng.choice(PRODUCTS), rng.choice(REGIONS)
    lo = rng.randint(1, 10)
    hi = rng.randint(lo, 12)
    intro = rng.choice([f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows).",
                        "sales.csv in this directory lists orders (one per row)."])
    kind = rng.choice(["units", "count", "revenue", "max", "best_region", "missing"])
    check, solvable = "exact_int", True
    if kind == "units":
        q = f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?"
        ans = sum(x["units"] for x in rows if x["product"] == p and lo <= x["month"] <= hi)
    elif kind == "count":
        q = f"How many orders came from the {r} region?"
        ans = sum(x["region"] == r for x in rows)
    elif kind == "revenue":
        q = f"What is the total revenue (units times price) of '{p}' orders?"
        ans = sum(x["units"] * x["price"] for x in rows if x["product"] == p)
    elif kind == "max":
        q = f"What is the largest number of units in a single order from the {r} region?"
        ans = max((x["units"] for x in rows if x["region"] == r), default=0)
    elif kind == "best_region":
        totals = {g: sum(x["units"] for x in rows if x["region"] == g and x["product"] == p) for g in REGIONS}
        best = max(totals, key=totals.get)
        if sorted(totals.values())[-1] == sorted(totals.values())[-2]:
            rows.append({"order_id": n + 1, "region": best, "product": p, "month": 1, "units": 7, "price": 9})
        q, ans, check = f"Which region sold the most units of '{p}'? Answer with the region name.", best, "exact_str"
    else:
        q, ans, solvable = f"What is the total discount given on '{p}' orders?", "", False
    if dirty:   # same products in mixed case and with stray spaces; the expected answer counts them all
        for x in rows:
            x["product"] = rng.choice([x["product"], x["product"].title(), x["product"] + " ", x["product"].upper()])
    d = emit(out, "data_analysis", i, f"{intro} {q}", ans, check, rng, solvable)
    (d / "workspace").mkdir()
    with open(d / "workspace" / "sales.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- search -------------------------------------------------------------------

FIRST = ["Alice", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Nikolai",
         "Olga", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Novak", "Smith", "Haddad", "Wang", "Kowalski", "Petrov", "Sato", "Garcia", "Muller"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Medan", "Tbilisi", "Hanoi", "Osaka", "Lyon", "Porto", "Cusco", "Tampere", "Graz", "Windhoek"]


def gen_search(out, i, rng):
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], 27)
    people, absent = names[:26], names[26]
    comp = {c: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2020)} for c in COMPANIES}
    pers = {p: {"Born": rng.randint(1950, 2000), "Employer": rng.choice(COMPANIES),
                "Mentor": rng.choice([q for q in people if q != p])} for p in people}
    x = rng.choice(people)
    m = pers[x]["Mentor"]
    c = pers[x]["Employer"]
    kind = rng.choice(["born", "employer", "hq", "founded", "mentor_born", "mentor_hq", "staff", "absent", "nofield"])
    solvable = True
    if kind == "born":
        q, ans = rng.choice([f"In which year was {x} born?", f"What is {x}'s year of birth?"]), pers[x]["Born"]
    elif kind == "employer":
        q, ans = f"Which company does {x} work for?", c
    elif kind == "hq":
        q, ans = rng.choice([f"In which city is the headquarters of {x}'s employer?",
                             f"{x} works for a company. Where is that company based?"]), comp[c]["Headquarters"]
    elif kind == "founded":
        q, ans = f"When was the company employing {x} founded?", comp[c]["Founded"]
    elif kind == "mentor_born":
        q, ans = f"In which year was the mentor of {x} born?", pers[m]["Born"]
    elif kind == "mentor_hq":
        q, ans = (f"In which city is the headquarters of the company that {x}'s mentor works for?",
                  comp[pers[m]["Employer"]]["Headquarters"])
    elif kind == "staff":
        q, ans = f"How many people in the knowledge base work for {c}?", sum(v["Employer"] == c for v in pers.values())
    elif kind == "absent":
        q, ans, solvable = f"In which year was {absent} born?", "", False
    else:
        q, ans, solvable = f"In which year was {x} founded?", "", False
    intro = rng.choice(["The directory kb/ contains a small knowledge base of markdown files.",
                        "Under kb/ there are notes about people and companies, one markdown file each."])
    d = emit(out, "search", i, f"{intro} {q}", ans, "exact_str", rng, solvable)
    kb = d / "workspace" / "kb"
    kb.mkdir(parents=True)
    for kind_, cards in (("person", pers), ("company", comp)):
        for title, fields in cards.items():
            body = "\n".join(f"{k}: {v}" for k, v in fields.items())
            (kb / f"{kind_}_{title.lower().replace(' ', '_')}.md").write_text(f"# {title}\n\n{body}\n")


# --- coding -------------------------------------------------------------------
# name: (correct source, buggy source, visible assert, hidden asserts)

FUNCS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                      "m.reverse_words('c a b'), 'b a c'", ["m.reverse_words('x'), 'x'", "m.reverse_words('b a'), 'a b'"]),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
              "m.total([1, 2, 3]), 6", ["m.total([]), 0", "m.total([-1, 5]), 4"]),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                "m.maximum([3, 9, 2]), 9", ["m.maximum([-5, -2]), -2", "m.maximum([7]), 7"]),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
                "m.average([2, 4, 6]), 4", ["m.average([5]), 5", "m.average([1, 2]), 1.5"]),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                "def first_n(xs, n):\n    return xs[n:]\n",
                "m.first_n([5, 6, 7, 8], 3), [5, 6, 7]", ["m.first_n([1, 2], 0), []", "m.first_n([1, 2, 3], 2), [1, 2]"]),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                   "m.count_even([1, 2, 4, 6, 7]), 3", ["m.count_even([]), 0", "m.count_even([1, 3]), 0"]),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      "def is_palindrome(s):\n    return s != s[::-1]\n",
                      "m.is_palindrome('level'), True", ["m.is_palindrome('ab'), False", "m.is_palindrome(''), True"]),
    "factorial": ("def factorial(n):\n    r = 1\n    for k in range(2, n + 1):\n        r *= k\n    return r\n",
                  "def factorial(n):\n    r = 1\n    for k in range(2, n):\n        r *= k\n    return r\n",
                  "m.factorial(5), 120", ["m.factorial(0), 1", "m.factorial(3), 6"]),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
              "m.clamp(15, 0, 10), 10", ["m.clamp(-3, 0, 10), 0", "m.clamp(4, 0, 10), 4"]),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
                     "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch not in 'aeiou')\n",
                     "m.count_vowels('Banana'), 3", ["m.count_vowels('xyz'), 0", "m.count_vowels('AEI'), 3"]),
}


def test_file(asserts):
    body = "".join(f"    def test_{k}(self):\n        self.assertEqual({a})\n\n" for k, a in enumerate(asserts))
    return f"import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n{body}\nif __name__ == '__main__':\n    unittest.main()\n"


def gen_coding(out, i, rng):
    names = rng.sample(list(FUNCS), rng.randint(4, 6))
    buggy = set(rng.sample(names, rng.randint(1, 3)))
    mode = rng.choice(["plain", "plain", "untested", "contradiction"])
    visible = [FUNCS[n][2] for n in names]
    solvable = True
    if mode == "untested":            # trap: one bug has no visible test, hidden tests still check it
        visible = [FUNCS[n][2] for n in names if n != sorted(buggy)[0]]
    elif mode == "contradiction":     # unsolvable: two tests demand different results for one call
        call = FUNCS[names[0]][2].rsplit(", ", 1)[0]
        visible.append(f"{call}, 'impossible'")
        solvable = False
    instruction = rng.choice([
        "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
        "(do not edit the tests) until all tests pass. When done, the answer is the single word DONE.",
        "mylib.py has defects. Repair it so that test_mylib.py passes; the tests themselves must stay as they are. "
        "Report DONE when finished.",
    ])
    d = emit(out, "coding", i, instruction, "tests_pass", "unittest", rng, solvable)
    (d / "workspace").mkdir()
    (d / "workspace" / "mylib.py").write_text("\n\n".join(FUNCS[n][n in buggy] for n in names))
    (d / "workspace" / "test_mylib.py").write_text(test_file(visible))
    (d / "hidden_tests").mkdir()
    hidden = [a for n in names for a in (FUNCS[n][2], *FUNCS[n][3])]
    (d / "hidden_tests" / "test_hidden.py").write_text(test_file(hidden))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    for gen in (gen_math, gen_data, gen_search, gen_coding):
        for i in range(args.per_type):
            gen(out, i, rng)
    print(f"{4 * args.per_type} tasks in {out}")


if __name__ == "__main__":
    main()
