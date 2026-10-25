import random

import ledger
from reliability import Faults, Payments, retrying, run_durable
from refunds import SYSTEM, grade, make_task, make_tools

task = make_task(7)
faults = Faults(seed=7, transient=0.15, timeout=0.15, duplicate=0.10)
backend = Payments(task["orders"], dedupe=True)   # stores the first reply for every key


def wrap(fn, name):  # a flaky network under each tool, then harness retries on top
    flaky = faults.wrap(fn, name, key_arg="order_id", effectful=(name == "refund"))
    return retrying(flaky, rng=random.Random(7))


r = run_durable(task["question"], make_tools(backend, "call_id", wrap), SYSTEM,
                part="a12", client=ledger.client())
print(faults.log)
print({k: v for k, v in grade(task, backend, r.answer).items() if k != "claimed"},
      f"model calls={len(r.steps)} cost=${r.cost:.5f}")
