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
# Second opinion. A wrong program can still run and print a plausible value,
# and no check of the harness sees that. One run uses well under a tenth of the
# budget, so part of the rest buys an independent solution of the answers
# marked risky (see pot.risky): the reasoning model solves the task from
# scratch, and where the two answers differ it is shown both programs and
# settles the dispute.
VERIFY = os.environ.get("HARNESS_VERIFY", "1") != "0"
VERIFY_STEP = (REASON, 2000)
VERIFY_BUDGET_SHARE = 0.5     # second opinions stop once this share of the budget is spent

DEFAULT_BUDGET = 100_000      # used only when BUDGET is not set
BUDGET_RESERVE = 0.03         # share kept back for counter drift
MIN_OUTPUT_TOKENS = 48        # a call that cannot afford this much output is not made
MAX_PARALLEL = 3              # the proxy key allows 3 calls
RUN_DEADLINE_S = 50 * 60      # stop starting new work; hard limit is 1 hour
LLM_TIMEOUT_S = 120
RATE_LIMIT_WAIT_S = 300      # HTTP 429 is retried this long; it costs no tokens
EXEC_TIMEOUT_S = 20           # generated program
TEST_TIMEOUT_S = 30           # task-provided unit tests

# Expected cost per task type, measured on the training runs; orders tasks by reward per token.
EXPECTED_COST = {"math": 67, "data_analysis": 62, "search": 61, "coding": 85}
DEFAULT_EXPECTED_COST = 250
