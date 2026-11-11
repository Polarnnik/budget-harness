from harness.solvers.coding import apply_edits, contradictory, parse_edits

SRC = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"


def test_edit_replaces_line_and_restores_lost_indentation():
    assert apply_edits(SRC, {4: "s += x"}).splitlines()[3] == "        s += x"


def test_empty_edit_deletes_line_and_out_of_range_is_ignored():
    assert apply_edits("a = 1\nb = 2\n", {2: "", 9: "zzz"}) == "a = 1\n"


def test_parse_edits_reads_numbers_headers_and_skips_fences():
    reply = "```\n4|         s += x\n=== util.py ===\n2: return 1\n```"
    assert parse_edits(reply, "mylib.py") == {"mylib.py": {4: "        s += x"}, "util.py": {2: "return 1"}}


def test_contradictory_tests_detected():
    body = "class T:\n def a(self):\n  self.assertEqual(m.f(1), 2)\n def b(self):\n  self.assertEqual(m.f(1), {})\n"
    assert contradictory(body.format(3))
    assert not contradictory(body.format(2))


def test_cut_off_reply_loses_only_its_last_line():
    reply = "3|     s = 1\n7|     return"
    assert parse_edits(reply, "m.py", truncated=True) == {"m.py": {3: "    s = 1"}}


def test_done_word_comes_from_the_instruction():
    from harness.solvers.coding import done_word
    assert done_word("Fix it. Then write ONLY the word FIXED to answer.txt.") == "FIXED"
    assert done_word("Repair the module and report when finished.") == "DONE"


def test_undecided_lists_only_swaps_the_tests_cannot_judge(tmp_path):
    import asyncio
    from pathlib import Path
    from harness.solvers import mutate
    work = tmp_path / "x" / "workspace"
    work.mkdir(parents=True)
    (work / "mylib.py").write_text(
        "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n\n\n"
        "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n")
    (work / "test_mylib.py").write_text(
        "import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        "    def test_a(self):\n        self.assertEqual(m.reverse_words('b a'), 'a b')\n"
        "    def test_b(self):\n        self.assertEqual(m.total([1, 2]), 3)\n\n\nunittest.main()\n")
    before = (work / "mylib.py").read_text()
    found = asyncio.run(mutate.undecided(work, Path("mylib.py"), [Path("test_mylib.py")]))
    assert found == [(2, "    return ' '.join(reversed(s.split()))")]     # `total` is pinned by its test
    assert (work / "mylib.py").read_text() == before


def test_slow_test_suite_skips_swap_analysis(tmp_path):
    """Tests that sleep must not be re-run once per swap: that would cost minutes."""
    import asyncio, time
    from pathlib import Path
    from harness.solvers import mutate
    work = tmp_path / "x" / "workspace"
    work.mkdir(parents=True)
    (work / "mylib.py").write_text("def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n")
    (work / "test_mylib.py").write_text(
        "import time, unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        "    def test_a(self):\n        time.sleep(2.5)\n        self.assertEqual(m.total([1, 2]), 3)\n\n\nunittest.main()\n")
    started = time.monotonic()
    assert asyncio.run(mutate.repair(work, Path("mylib.py"), [Path("test_mylib.py")], 30)) is False
    assert asyncio.run(mutate.undecided(work, Path("mylib.py"), [Path("test_mylib.py")])) == []
    assert time.monotonic() - started < 9          # two baseline runs, no per-swap runs
    assert "s -= x" in (work / "mylib.py").read_text()
