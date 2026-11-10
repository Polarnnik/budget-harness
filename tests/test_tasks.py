from harness.tasks import Task, infer_type, missing_input


def make(tmp_path, kind, files=()):
    for name in files:
        f = tmp_path / "workspace" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    return Task("t", tmp_path, kind, "", 1, "")


def test_declared_type_wins_over_stray_files(tmp_path):
    assert infer_type(make(tmp_path, "math", ["notes.md", "more.md"])) == "math"


def test_unknown_type_is_guessed_from_files(tmp_path):
    assert infer_type(make(tmp_path, "", ["sales.csv"])) == "data_analysis"
    assert infer_type(make(tmp_path / "b", "puzzle", [])) == "math"


def test_missing_input_only_for_declared_family_without_files(tmp_path):
    assert missing_input(make(tmp_path / "a", "data_analysis", ["readme.md"]))
    assert not missing_input(make(tmp_path / "b", "data_analysis", ["sales.csv"]))
    assert not missing_input(make(tmp_path / "c", "math", []))
