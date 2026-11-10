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
    llm = FakeLLM(dict(REPLIES, **{"s += x + 1": "2: return ' '.join(reversed(s.split()))\n8: s += x"}))
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
