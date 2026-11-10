import asyncio

from harness.llm import LLM, Budget


class Usage:
    input_tokens, output_tokens, total_tokens, reasoning_tokens = 200, 30, 230, 0


class Resp:
    content, usage_metadata = "42", Usage


class SlowModel:
    def __init__(self):
        self.caps = []

    async def invoke(self, messages, model, max_tokens, temperature):
        self.caps.append(max_tokens)
        await asyncio.sleep(0.01)
        return Resp


def make(total):
    llm = LLM.__new__(LLM)        # skip openJiuwen client construction
    llm.budget, llm.trace_path, llm.calls, llm.failures_in_row, llm.dead = Budget(total, 0), None, 0, 0, False
    llm._gate, llm._model = asyncio.Semaphore(5), SlowModel()
    return llm


MSG = [{"role": "user", "content": "x" * 600}]      # about 200 input tokens


def test_parallel_calls_wait_for_reservations_instead_of_failing():
    """Two tasks with a small budget: holds of 256 cannot overlap, yet both calls run."""
    llm = make(400)

    async def go():
        return await asyncio.gather(*(llm.ask("m", MSG, 256) for _ in range(2)))

    replies = asyncio.run(go())
    assert all(r is not None for r in replies)
    assert llm.budget.spent == 100 and llm.budget.held == 0


def test_output_cap_shrinks_to_remaining_budget():
    llm = make(150)
    assert asyncio.run(llm.ask("m", MSG, 256)) is not None
    assert llm._model.caps == [150 - 22]        # 600 chars -> 216 estimated input tokens


def test_call_refused_when_budget_cannot_cover_minimum_output():
    llm = make(60)
    assert asyncio.run(llm.ask("m", MSG, 256)) is None
    assert llm._model.caps == [] and llm.budget.spent == 0
