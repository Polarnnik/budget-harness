import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.llm import Budget, Reply  # noqa: E402


class FakeLLM:
    """Scripted model: the first rule whose key occurs in the prompt answers."""

    def __init__(self, rules: dict[str, list[str] | str], budget: float = 10_000):
        self.rules = {k: [v] if isinstance(v, str) else list(v) for k, v in rules.items()}
        self.budget = Budget(budget)
        self.calls = 0
        self.prompts: list[str] = []

    async def ask(self, model, messages, max_tokens, tag=""):
        prompt = "\n".join(m["content"] for m in messages)
        self.prompts.append(prompt)
        self.calls += 1
        for key, replies in self.rules.items():
            if key in prompt:
                text = replies.pop(0) if len(replies) > 1 else replies[0]
                return Reply(text, len(prompt) // 3, len(text) // 3, 0)
        raise AssertionError(f"no scripted reply for prompt:\n{prompt}")


@pytest.fixture
def tasks_dir(tmp_path):
    """Private copy of the organisers' example tasks."""
    dst = tmp_path / "tasks"
    shutil.copytree(ROOT / "examples" / "tasks", dst)
    return dst
