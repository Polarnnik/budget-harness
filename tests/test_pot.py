from harness.solvers.pot import extract_code


def test_plain_expression_passes_through():
    assert extract_code("1669 - 36*3") == "1669 - 36*3"


def test_code_fence_is_removed():
    assert extract_code("```python\nx = 1\nx + 1\n```") == "x = 1\nx + 1"


def test_prose_around_code_is_dropped():
    assert extract_code("Here is the code:\nx = 2\nx * 3\nThis gives 6.") == "x = 2\nx * 3"


def test_broken_code_is_returned_for_the_retry_message():
    assert extract_code("1 +") == "1 +"

