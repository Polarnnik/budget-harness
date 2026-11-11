"""End-to-end on the organisers' eight example tasks with a scripted model."""
import asyncio
import json
import shutil
import subprocess
import sys
import time

from conftest import FakeLLM
from harness import agent

REPLIES = {
    "1752 boxes": "x = 1752\nx //= 2\nx *= 2\nx //= 3\nx *= 2\nx // 3",
    "1669 boxes": "1669 - 36*3 + 2*87",
    "months 7 through 8": "sales[(sales['product']=='widget')&sales['month'].between(7,8)]['units'].sum()",
    "months 5 through 6": "```python\nsales[(sales['product']=='widget')&sales['month'].between(5,6)]['units'].sum()\n```",
    "Irina Smith": "kb['Irina Smith']['Born']",
    "Sergei Garcia": "kb[kb['sergei garcia']['Employer']]['Headquarters']",
    "All shown tests pass": "OK",
}


def run(tasks_dir, llm):
    return asyncio.run(agent.run(tasks_dir, llm, time.monotonic() + 600))


def test_all_examples_solved(tasks_dir):
    llm = FakeLLM(REPLIES)
    status = run(tasks_dir, llm)
    assert set(status.values()) == {"answered"}
    assert llm.calls == 8                       # one call per task
    for d in tasks_dir.iterdir():
        spec = json.loads((d / "task.json").read_text())
        answer = (d / "answer.txt").read_text().strip()
        if spec["task_type"] == "math":
            assert not (d / "workspace").exists()        # never created just to hold a copy
        else:
            assert (d / "workspace" / "answer.txt").read_text().strip() == answer
        if spec["check"] == "unittest":
            assert answer == "DONE"
            r = subprocess.run([sys.executable, "test_mylib.py"], cwd=d / "workspace", capture_output=True)
            assert r.returncode == 0, r.stderr.decode()
        else:
            assert answer == spec["expected"], d.name


def test_prompts_carry_no_raw_data(tasks_dir):
    llm = FakeLLM(REPLIES)
    run(tasks_dir, llm)
    data = next(p for p in llm.prompts if "months 5 through 6" in p)
    assert "300 rows" in data and len(data) < 2500      # schema, not the rows
    search = next(p for p in llm.prompts if "Sergei Garcia" in p)
    assert "kb['Sergei Garcia'] = {'Born': 1985" in search   # the asked card is the sample


def test_retry_after_failed_program(tasks_dir):
    replies = dict(REPLIES, **{"1669 boxes": ["1669 - 36*3 + 2*87 +", "1669 - 36*3 + 2*87"]})
    llm = FakeLLM(replies)
    run(tasks_dir, llm)
    assert (tasks_dir / "math_0004" / "answer.txt").read_text().strip() == "1735"
    assert "SyntaxError" in next(p for p in llm.prompts if "That failed" in p)


def test_missing_card_stops_without_retry(tasks_dir):
    llm = FakeLLM(dict(REPLIES, **{"Irina Smith": "kb['Zed Nobody']['Born']"}))
    status = run(tasks_dir, llm)
    assert status["search_0001"] == "no answer"
    assert llm.calls == 8


def not_a_swap(tasks_dir):
    """Give coding_0003 a bug no operator swap undoes, so the model has to fix it."""
    src = tasks_dir / "coding_0003" / "workspace" / "mylib.py"
    src.write_text(src.read_text().replace("s -= x", "s += x + 1"))
    return src


def test_model_patch_applied_when_swaps_do_not_help(tasks_dir):
    src = not_a_swap(tasks_dir)
    llm = FakeLLM(dict(REPLIES, **{"s += x + 1": "2| return ' '.join(reversed(s.split()))\n8| s += x"}))
    run(tasks_dir, llm)
    assert "        s += x\n" in src.read_text()
    r = subprocess.run([sys.executable, "test_mylib.py"], cwd=src.parent, capture_output=True)
    assert r.returncode == 0


def test_failed_patch_leaves_source_untouched(tasks_dir):
    src = not_a_swap(tasks_dir)
    before = src.read_text()
    llm = FakeLLM(dict(REPLIES, **{"s += x + 1": ["8: s *= x", "8: s *= x"]}))
    run(tasks_dir, llm)
    assert src.read_text() == before
    assert not (tasks_dir / "coding_0003" / "answer.txt").exists()     # tests fail: no answer file


def test_budget_exhausted_writes_nothing(tasks_dir):
    class Broke(FakeLLM):
        async def ask(self, *a, **k):
            return None
    status = run(tasks_dir, Broke({}))
    assert status["math_0000"] == "no answer"
    assert not (tasks_dir / "math_0000" / "answer.txt").exists()


def test_answer_written_by_the_program_is_used(tasks_dir):
    llm = FakeLLM(dict(REPLIES, **{"1669 boxes": "open('answer.txt', 'w').write(str(1669 - 36*3 + 2*87))\nNone"}))
    run(tasks_dir, llm)
    assert (tasks_dir / "math_0004" / "answer.txt").read_text().strip() == "1735"


def test_review_edit_that_breaks_tests_is_dropped(tasks_dir):
    llm = FakeLLM(dict(REPLIES, **{"All shown tests pass": "1: def broken(:"}))
    run(tasks_dir, llm)
    ws = tasks_dir / "coding_0007" / "workspace"
    assert subprocess.run([sys.executable, "test_mylib.py"], cwd=ws, capture_output=True).returncode == 0


def test_task_without_its_input_files_costs_nothing(tasks_dir):
    shutil.rmtree(tasks_dir / "search_0001" / "workspace" / "kb")
    llm = FakeLLM(REPLIES)
    status = run(tasks_dir, llm)
    assert status["search_0001"] == "no answer"
    assert not any("Irina Smith" in p for p in llm.prompts)


def test_missing_field_stops_without_escalation(tasks_dir):
    llm = FakeLLM(dict(REPLIES, **{"Irina Smith": "kb['Irina Smith']['Founded']"}))
    status = run(tasks_dir, llm)
    assert status["search_0001"] == "no answer" and llm.calls == 8


def test_missing_column_stops_but_a_typo_gets_a_retry(tasks_dir):
    llm = FakeLLM(dict(REPLIES, **{
        "months 7 through 8": "sales['discount'].sum()",
        "months 5 through 6": ["sales['unit'].sum()", "sales['units'].sum()"]}))
    status = run(tasks_dir, llm)
    assert status["data_analysis_0002"] == "no answer"
    assert status["data_analysis_0006"] == "answered" and llm.calls == 9


def test_contradictory_tests_leave_no_answer(tasks_dir):
    t = tasks_dir / "coding_0007" / "workspace" / "test_mylib.py"
    t.write_text(t.read_text().replace(
        "    def test_total", "    def test_x(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 7)\n\n    def test_total"))
    status = run(tasks_dir, FakeLLM(REPLIES))
    assert status["coding_0007"] == "no answer"


def test_prose_knowledge_base_goes_to_the_model_as_notes(tasks_dir):
    kb = tasks_dir / "search_0001" / "workspace" / "kb"
    for f in kb.iterdir():
        f.write_text(f.read_text().replace(": ", " is "))        # no `Field: value` lines left
    llm = FakeLLM(dict(REPLIES, **{"Irina Smith": "1983"}))
    run(tasks_dir, llm)
    prompt = next(p for p in llm.prompts if "Irina Smith" in p)
    assert "Notes from the working directory" in prompt and "Born is 1983" in prompt
    assert (tasks_dir / "search_0001" / "answer.txt").read_text().strip() == "1983"


def test_second_step_builds_on_partial_progress_with_fresh_numbers(tasks_dir):
    """Step one fixes one of two bugs; step two sees that state and fixes the other."""
    src = not_a_swap(tasks_dir)
    src.write_text(src.read_text().replace("sorted(s.split())", "list(s.split())[:1]"))
    llm = FakeLLM(dict(REPLIES, **{"def first_n": ["8|         s += x", "2|     return ' '.join(reversed(s.split()))"]}))
    run(tasks_dir, llm)
    assert subprocess.run([sys.executable, "test_mylib.py"], cwd=src.parent, capture_output=True).returncode == 0
    first, second = [p for p in llm.prompts if "def first_n" in p]
    assert "8|         s += x + 1" in first and "test_total" in first.split("Failing now:")[1]
    assert "8|         s += x\n" in second and "test_total" not in second.split("Failing now:")[1]


def answer(tasks_dir, task):
    return (tasks_dir / task / "answer.txt").read_text().strip()


DROPPED = "1669 - 36*3"            # leaves out the deliveries: the numbers 2 and 87 are unused
FULL = "1669 - 36*3 + 2*87"


def test_plain_answers_get_no_second_opinion(tasks_dir):
    """The organisers' examples use every number and plain lookups: nothing is risky, nothing is re-solved."""
    llm = FakeLLM(REPLIES)
    run(tasks_dir, llm)
    assert all(t.split("#")[1].isdigit() for t in llm.tags)


def test_dropped_step_gets_a_second_opinion_and_the_dispute_is_settled(tasks_dir):
    from harness import config
    llm = FakeLLM(dict(REPLIES, **{"1669 boxes": DROPPED}), by_model={("1669 boxes", config.REASON): FULL})
    status = run(tasks_dir, llm)
    assert answer(tasks_dir, "math_0004") == "1735"
    assert status["math_0004"] == "answered, changed by second opinion"
    dispute = next(p for p in llm.prompts if "Two programs for this task disagree" in p)
    assert f"Program A gives '1561':\n{DROPPED}\n" in dispute and "Program B gives '1735'" in dispute


def test_dispute_settled_for_the_first_answer_keeps_it(tasks_dir):
    from harness import config
    llm = FakeLLM(dict(REPLIES, **{"1669 boxes": DROPPED}), by_model={
        ("Two programs for this task disagree", config.REASON): DROPPED, ("1669 boxes", config.REASON): FULL})
    status = run(tasks_dir, llm)
    assert answer(tasks_dir, "math_0004") == "1561" and status["math_0004"] == "answered"
    assert "math_0004#d0" in llm.tags


def test_risky_marks_aggregates_and_notes_but_not_plain_lookups():
    from harness.solvers.pot import risky
    assert not risky("q", "search", "kb[kb['Sergei Garcia']['Employer']]['Headquarters']")
    assert risky("q", "search", "min(kb, key=lambda n: kb[n]['Born'])")
    assert risky("q", "notes", "'Osaka'")
    assert not risky("A shop has 5 apples and sells 2. Write it: echo 42 > answer.txt.", "math", "5 - 2")
    assert not risky("q", "data_analysis", "sales['units'].sum()")
