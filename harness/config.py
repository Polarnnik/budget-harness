"""Tunable constants. Token costs are in output-token equivalents (10 input = 1 output)."""
import os

FAST = "qwen3.8-27b-noreason"
REASON = "qwen3.8-27b"

# Escalation chain: (model, max_tokens). The next step runs only when the
# harness-side check of the previous answer fails. Measured on 60 training
# tasks, both models alone: FAST solves 46 of 47 at 63 tokens per task, REASON
# solves 47 of 47 at 226. Reasoning is billed as output, so it is the fallback
# and its cap is hard.
CHAIN = [(FAST, 256), (REASON, 2000)]
# Experiments only (tools/compare_models.py): HARNESS_CHAIN="model:max_tokens,model:max_tokens".
if os.environ.get("HARNESS_CHAIN"):
    CHAIN = [(m, int(n)) for m, n in (step.rsplit(":", 1) for step in os.environ["HARNESS_CHAIN"].split(","))]
DEFAULT_BUDGET = 100_000      # used only when BUDGET is not set
BUDGET_RESERVE = 0.03         # share kept back for counter drift
MIN_OUTPUT_TOKENS = 48        # a call that cannot afford this much output is not made
MAX_PARALLEL = 5              # sessions allowed by the rules
RUN_DEADLINE_S = 50 * 60      # stop starting new work; hard limit is 1 hour
LLM_TIMEOUT_S = 120
RATE_LIMIT_RETRIES = 8        # HTTP 429 is retried with backoff; it costs no tokens
EXEC_TIMEOUT_S = 20           # generated program
TEST_TIMEOUT_S = 30           # task-provided unit tests

# Rough expected cost per task type, used only to order tasks by reward per token.
EXPECTED_COST = {"math": 60, "data_analysis": 80, "search": 70, "coding": 120}
DEFAULT_EXPECTED_COST = 250
