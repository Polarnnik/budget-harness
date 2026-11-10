"""Run the harness once per model on the same task sets and print a comparison.

    LLM_BASE_URL=... LLM_API_KEY=... python tools/compare_models.py --out runs/cmp SET [SET ...]

Each model works alone (a one-step chain, no escalation), so the numbers
describe the model, not the cascade. Reasoning models get a larger output cap:
their thinking is billed as output and a tight cap cuts them off mid-thought.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import ROOT, passed  # noqa: E402

MODELS = {"qwen3.6-35b-a3b-noreason": 256, "qwen3.8-27b-noreason": 256,
          "qwen3.6-35b-a3b": 2000, "qwen3.8-27b": 2000}


def run(model: str, cap: int, tasks_src: Path, out: Path) -> None:
    tasks = out / "tasks"
    shutil.copytree(tasks_src, tasks)
    env = dict(os.environ, BUDGET="1000000", HARNESS_CHAIN=f"{model}:{cap}", HARNESS_TRACE=str(out / "trace.jsonl"))
    with open(out / "run.log", "w") as log:
        subprocess.run([sys.executable, "-B", str(ROOT / "run_agent.py"), "--tasks-dir", str(tasks)],
                       env=env, stdout=log, stderr=subprocess.STDOUT)


def summarise(out: Path) -> dict:
    calls = defaultdict(list)
    for line in (out / "trace.jsonl").read_text().splitlines() if (out / "trace.jsonl").is_file() else []:
        row = json.loads(line)
        calls[row.get("tag", "").split("#")[0]].append(row)
    stats = defaultdict(lambda: defaultdict(float))
    seconds = []
    for d in sorted((out / "tasks").iterdir()):
        if not (d / "task.json").is_file():
            continue
        spec = json.loads((d / "task.json").read_text())
        ok = passed(d, spec)
        rows = calls[spec["task_id"]]
        seconds += [r["seconds"] for r in rows if "seconds" in r]
        for key in (spec["task_type"], "ALL"):
            s = stats[key]
            solvable = spec.get("solvable", True)
            s["n"] += 1
            s["solvable"] += solvable
            s["solved"] += ok and solvable
            s["reward"] += spec.get("reward", 0) * (ok and solvable)
            s["cost"] += sum(r.get("cost", 0) for r in rows)
            s["cost_unsolvable"] += 0 if solvable else sum(r.get("cost", 0) for r in rows)
            s["in"] += sum(r.get("tokens_in", 0) for r in rows)
            s["out"] += sum(r.get("tokens_out", 0) for r in rows)
            s["reasoning"] += sum(r.get("reasoning", 0) or 0 for r in rows)
            s["errors"] += sum("error" in r for r in rows)
    seconds.sort()
    stats["ALL"]["sec_mean"] = statistics.mean(seconds) if seconds else 0
    stats["ALL"]["sec_p95"] = seconds[int(0.95 * (len(seconds) - 1))] if seconds else 0
    stats["ALL"]["sec_max"] = seconds[-1] if seconds else 0
    return {k: dict(v) for k, v in stats.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out).resolve()
    result = {}
    for src in map(Path, args.sets):
        for model, cap in MODELS.items():
            out = root / src.name / model
            if not out.exists():
                out.mkdir(parents=True)
                run(model, cap, src, out)
            result[f"{src.name}/{model}"] = summarise(out)
            a = result[f"{src.name}/{model}"]["ALL"]
            print(f"{src.name:10} {model:26} solved {a['solved']:.0f}/{a['solvable']:.0f} reward {a['reward']:.0f} "
                  f"cost {a['cost']:.0f} (in {a['in']:.0f}, out {a['out']:.0f}, reasoning {a['reasoning']:.0f}) "
                  f"sec mean {a['sec_mean']:.1f} p95 {a['sec_p95']:.1f} max {a['sec_max']:.1f} errors {a['errors']:.0f}",
                  flush=True)
    (root / "summary.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
