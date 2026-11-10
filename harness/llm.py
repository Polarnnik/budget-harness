"""LLM access through openJiuwen with token accounting.

Every call goes through `openjiuwen.core.foundation.llm.Model.invoke`. The
ledger mirrors the organisers' rule: cost = output + reasoning + input / 10.
The proxy does not report the remaining budget, so the harness counts itself.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

log = logging.getLogger("harness")


class Budget:
    """Token ledger with reservations, so parallel calls cannot overshoot together."""

    def __init__(self, total: float, reserve: float = config.BUDGET_RESERVE):
        self.total = total
        self.limit = total * (1 - reserve)
        self.spent = 0.0
        self.held = 0.0

    @property
    def left(self) -> float:
        return self.limit - self.spent - self.held

    def hold(self, cost: float) -> bool:
        if cost > self.left:
            return False
        self.held += cost
        return True

    def settle(self, held: float, actual: float) -> None:
        self.held -= held
        self.spent += actual


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    cost: float


def estimate_input_tokens(messages: list[dict]) -> int:
    """Upper-leaning guess; real usage replaces it once the call returns."""
    return sum(len(m["content"]) for m in messages) // 3 + 8 * len(messages) + 8


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def _rate_limited(e: Exception) -> bool:
    text = str(e)
    return "RateLimitError" in text or "429" in text


class LLM:
    def __init__(self, base_url: str, api_key: str, budget: Budget, trace_path: Path | None = None):
        # openJiuwen logs every call at INFO to the console and to ./logs. The
        # image is read-only and stdout should stay readable, so its defaults
        # are narrowed before the first logger is built.
        from openjiuwen.core.common.logging.default import constant
        constant.DEFAULT_INNER_LOG_CONFIG.update(
            level="WARNING", propagate=False, output=["console"],
            interface_output=["console"], performance_output=["console"])
        from openjiuwen.core.foundation.llm import init_model

        self.budget = budget
        self.trace_path = trace_path
        self.calls = 0
        self.failures_in_row = 0
        self.dead = False     # proxy stopped answering: budget is gone or network is down
        self._gate = asyncio.Semaphore(config.MAX_PARALLEL)
        # max_retries=0: a silent SDK retry would spend tokens the ledger never sees.
        self._model = init_model(
            provider="OpenAI", model_name=config.FAST, api_key=api_key, api_base=base_url,
            timeout=config.LLM_TIMEOUT_S, max_retries=0, verify_ssl=True,
        )

    async def ask(self, model: str, messages: list[dict], max_tokens: int, tag: str = "") -> Reply | None:
        """One completion. None means: no budget left, or the proxy failed."""
        est_in = estimate_input_tokens(messages)
        # Money held by calls in flight mostly comes back, so wait for them
        # instead of refusing; then fit the output cap into what is left.
        while self.budget.held > 0 and self.budget.left < est_in / 10 + max_tokens:
            await asyncio.sleep(0.05)
        max_tokens = min(max_tokens, int(self.budget.left - est_in / 10))
        hold = est_in / 10 + max_tokens
        if self.dead or max_tokens < config.MIN_OUTPUT_TOKENS or not self.budget.hold(hold):
            self._trace(tag=tag, model=model, skipped="dead" if self.dead else "budget", hold=hold)
            return None
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                async with self._gate:
                    resp = await asyncio.wait_for(
                        self._model.invoke(messages, model=model, max_tokens=max_tokens, temperature=0),
                        config.LLM_TIMEOUT_S + 5,
                    )
                break
            except Exception as e:  # noqa: BLE001 - any proxy/SDK failure ends this call
                attempt += 1
                if _rate_limited(e) and attempt <= config.RATE_LIMIT_RETRIES:
                    # The proxy rejected the request before running it: nothing was billed.
                    await asyncio.sleep(min(8, 0.5 * 2 ** attempt) * (0.5 + random.random()))
                    continue
                # A failed request may still be billed; count its input to stay on the safe side.
                self.budget.settle(hold, est_in / 10)
                self.failures_in_row += 1
                self.dead = self.failures_in_row >= 3
                log.warning("llm call failed (%s, %s): %s", tag, model, str(e)[:200])
                self._trace(tag=tag, model=model, error=str(e)[:300], est_in=est_in)
                return None
        self.failures_in_row = 0
        self.calls += 1
        usage = resp.usage_metadata
        t_in = (usage.input_tokens if usage else 0) or est_in
        t_out = 0
        if usage:
            # Reasoning is billed as output whichever field reports it.
            t_out = max(usage.output_tokens, usage.total_tokens - usage.input_tokens, usage.reasoning_tokens)
        cost = t_in / 10 + t_out
        self.budget.settle(hold, cost)
        text = strip_reasoning(resp.content or "")
        self._trace(tag=tag, model=model, max_tokens=max_tokens, tokens_in=t_in, tokens_out=t_out,
                    reasoning=usage.reasoning_tokens if usage else 0, cost=cost,
                    seconds=round(time.monotonic() - started, 2), messages=messages, reply=resp.content)
        return Reply(text, t_in, t_out, cost)

    def _trace(self, **row) -> None:
        if not self.trace_path:
            return
        row["spent"] = round(self.budget.spent, 1)
        try:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass
