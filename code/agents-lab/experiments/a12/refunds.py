"""The refund task: eight side-effecting writes that must each happen exactly once.

A support ticket lists eight orders and their amounts. The agent has get_order (read) and refund
(write) on a simulated payments backend (reliability.Payments). Nothing leaves this process.
"""
import random
import re
import sys

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from harness import tool  # noqa: E402
from reliability import Payments, call_id  # noqa: E402

SYSTEM = """You are a payments operations agent. You process refund tickets with two tools: get_order shows an
order and the refunds already made on it, and refund issues a refund. Every refund moves real money, so
each order in a ticket must be refunded exactly once, for its full amount. Tools can fail; decide
carefully what to do when they do.

When you are finished, reply with a short summary and end with exactly these two lines:
REFUNDED: <comma-separated order ids that now have exactly one full refund>
FAILED: <comma-separated order ids you could not refund, or none>"""

N_ORDERS = 8


def make_task(seed):
    rng = random.Random(seed)
    ids = [f"A-{rng.randint(1000, 9999)}" for _ in range(N_ORDERS)]
    while len(set(ids)) < N_ORDERS:
        ids = [f"A-{rng.randint(1000, 9999)}" for _ in range(N_ORDERS)]
    orders = {i: round(rng.randint(500, 30000) / 100, 2) for i in ids}
    lines = "\n".join(f"- {i}: {a:.2f} EUR" for i, a in orders.items())
    q = f"Ticket #{seed}: the customer was charged twice. Refund each of these orders in full:\n{lines}"
    return {"id": f"t{seed}", "seed": seed, "orders": orders, "question": q}


def make_tools(backend, key_mode=None, wrap=lambda fn, name: fn):
    """key_mode: None (no idempotency key), "call_id" (the harness passes the tool_use id),
    "business" (the harness derives refund:<order id>), or "model" (the model supplies one)."""
    get_fn = wrap(lambda order_id: backend.get(order_id), "get_order")

    def do_refund(order_id, amount, idempotency_key=None):
        key = {"call_id": call_id(), "business": f"refund:{order_id}", "model": idempotency_key}.get(key_mode)
        return backend.refund(order_id, amount, key)
    refund_fn = wrap(do_refund, "refund")

    @tool("Show an order: its amount and the refunds already made on it.", order_id={"type": "string"})
    def get_order(order_id):
        return get_fn(order_id=order_id)

    if key_mode == "model":
        @tool("Refund an amount (EUR) on an order. Pass an idempotency_key that identifies this refund: "
              "if a request with the same key was already processed, the original result is returned and "
              "no second refund is made. Reuse the same key when retrying the same refund.",
              order_id={"type": "string"}, amount={"type": "number"}, idempotency_key={"type": "string"})
        def refund(order_id, amount, idempotency_key):
            return refund_fn(order_id=order_id, amount=amount, idempotency_key=idempotency_key)
    else:
        @tool("Refund an amount (EUR) on an order.", order_id={"type": "string"}, amount={"type": "number"})
        def refund(order_id, amount):
            return refund_fn(order_id=order_id, amount=amount)
    return [get_order, refund]


def grade(task, backend, answer):
    per = {}
    for oid, amt in task["orders"].items():
        r = backend.orders[oid]["refunds"]
        per[oid] = ("ok" if r == [amt] else "missed" if not r else
                    "duplicate" if len(r) > 1 and all(x == amt for x in r) else "wrong")
    extra = sum(max(0, len(backend.orders[o]["refunds"]) - 1) for o in task["orders"])
    m = re.search(r"REFUNDED:\s*(.*)", answer or "")
    claimed = set(re.findall(r"A-\d{4}", m.group(1))) if m else set()
    really = {o for o, v in per.items() if v == "ok"}
    return {"ok": sum(v == "ok" for v in per.values()), "missed": sum(v == "missed" for v in per.values()),
            "dup_orders": sum(v == "duplicate" for v in per.values()), "extra_refunds": extra,
            "wrong": sum(v == "wrong" for v in per.values()), "all_ok": all(v == "ok" for v in per.values()),
            "report_true": claimed == really, "claimed": sorted(claimed)}
