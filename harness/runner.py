"""Child-process entry: run a generated program and print its answer.

Usage: python -I -B runner.py program.py prelude.py   (cwd = workspace copy)

The program is treated like a REPL cell: if its last statement is an
expression, the value is printed. That lets the model answer with a bare
expression, which is the cheapest possible output.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for `harness.kb`


def show(v):
    """Normalise a result to the answer-file form."""
    if hasattr(v, "item") and getattr(v, "shape", None) == ():   # numpy scalar
        v = v.item()
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):  # Series / ndarray
        v = v.tolist()
    if isinstance(v, (list, tuple, set, frozenset)):
        items = list(v)
        if len(items) == 1:
            return show(items[0])
        return ", ".join(show(x) for x in items)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def main():
    program, prelude = (Path(p).read_text(encoding="utf-8") for p in sys.argv[1:3])
    env = {"__name__": "__main__"}
    exec(compile(prelude, "prelude", "exec"), env)
    tree = ast.parse(program, "program")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "program", "exec"), env)
    if last is not None:
        value = eval(compile(last, "program", "eval"), env)
        if value is not None:
            print(show(value))


if __name__ == "__main__":
    main()
