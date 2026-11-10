from harness.solvers.coding import apply_edits, contradictory, parse_edits

SRC = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"


def test_edit_replaces_line_and_restores_lost_indentation():
    assert apply_edits(SRC, {4: "s += x"}).splitlines()[3] == "        s += x"


def test_empty_edit_deletes_line_and_out_of_range_is_ignored():
    assert apply_edits("a = 1\nb = 2\n", {2: "", 9: "zzz"}) == "a = 1\n"


def test_parse_edits_reads_numbers_headers_and_skips_fences():
    reply = "```\n4:         s += x\n# util.py\n2: return 1\n```"
    assert parse_edits(reply, "mylib.py") == {"mylib.py": {4: "        s += x"}, "util.py": {2: "return 1"}}


def test_contradictory_tests_detected():
    body = "class T:\n def a(self):\n  self.assertEqual(m.f(1), 2)\n def b(self):\n  self.assertEqual(m.f(1), {})\n"
    assert contradictory(body.format(3))
    assert not contradictory(body.format(2))
