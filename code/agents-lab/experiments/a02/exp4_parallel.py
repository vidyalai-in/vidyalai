"""Exp 4: parallel tool calls vs disable_parallel_tool_use on a 20-task set (Haiku 5.5).

usage: python exp4_parallel.py <repeats> [n_tasks] [arms]
"""
import sys, time, random, json, hashlib
from concurrent.futures import ThreadPoolExecutor
from common import ledger, HAIKU, dump
from toollib import TOOLS

cl = ledger.client()
NAMES = ["get_weather", "get_stock_price", "convert_currency", "get_order", "check_inventory", "get_shipping_quote"]
TOOLSET = [t for t in TOOLS if t["name"] in NAMES]
SYSTEM = "You are a helpful assistant. Use the tools to get facts; answer in two or three sentences."


def h(*xs):
    return int(hashlib.md5(json.dumps(xs).encode()).hexdigest()[:6], 16)


def run_tool(name, a):
    """Deterministic fake backends."""
    if name == "get_weather":
        return f"{a.get('city')}: {5 + h(a.get('city')) % 25} C, {['sunny', 'cloudy', 'light rain'][h(a.get('city')) % 3]}"
    if name == "get_stock_price":
        return f"{a.get('ticker')}: ${50 + h(a.get('ticker')) % 900}.{h(a.get('ticker')) % 100:02d}"
    if name == "convert_currency":
        rate = 0.5 + (h(a.get('from_currency'), a.get('to_currency')) % 150) / 100
        return f"{a.get('amount')} {a.get('from_currency')} = {round(float(a.get('amount', 0)) * rate, 2)} {a.get('to_currency')} (rate {rate})"
    if name == "get_order":
        oid = a.get("order_id")
        return json.dumps({"order_id": oid, "status": "shipped", "sku": f"SKU-{h(oid) % 900 + 100}",
                           "total_usd": 20 + h(oid) % 300, "ship_to_postal": f"{10000 + h(oid) % 89999}"})
    if name == "check_inventory":
        return f"{a.get('sku')}: {h(a.get('sku')) % 40} units in Reno, {h(a.get('sku'), 1) % 40} in Newark"
    if name == "get_shipping_quote":
        return f"${4 + h(a) % 20}.99, {1 + h(a) % 6} days"
    return f"error: unknown tool {name}"


# (kind, prompt, minimum number of model turns that issue tool calls)
TASKS = [
    ("indep", "What's the weather in Paris, London and Tokyo right now?", 1),
    ("indep", "Give me the current prices of AAPL, MSFT and NVDA.", 1),
    ("indep", "How's the weather in Berlin, and what's TSLA trading at?", 1),
    ("indep", "Check stock for SKU-101, SKU-202, SKU-303 and SKU-404.", 1),
    ("indep", "Convert 100 USD to EUR, to GBP and to JPY.", 1),
    ("indep", "What's the status of orders ORD-1001 and ORD-1002?", 1),
    ("indep", "Weather in Mumbai and Delhi, and the price of INFY.", 1),
    ("indep", "Quote shipping for a 2000 g parcel from 94105 to 10001, and from 94105 to 60601.", 1),
    ("indep", "Compare the weather in Sydney and Melbourne.", 1),
    ("indep", "I need the prices of GOOG and AMZN, and 250 EUR converted to USD.", 1),
    ("indep", "Look up orders ORD-2001, ORD-2002 and ORD-2003.", 1),
    ("indep", "How much stock do we have of SKU-555 and SKU-777, and what is the weather in Reno?", 1),
    ("indep", "Price of AAPL, and the weather in Cupertino and Seattle.", 1),
    ("indep", "Convert 50 GBP to USD and 50 GBP to EUR, and get the price of BP.", 1),
    ("dep", "Look up order ORD-3001 and tell me how much inventory we have of the SKU in it.", 2),
    ("dep", "What's NVDA's price in euros?", 2),
    ("dep", "Get order ORD-4002, then quote shipping a 1500 g replacement from 94105 to that order's postal code.", 2),
    ("dep", "Find the total of order ORD-5003 and convert it to JPY.", 2),
    ("mixed", "Check inventory for the SKU in order ORD-6004, and also tell me the weather in Newark.", 2),
    ("mixed", "Get the price of MSFT in GBP, and the weather in London.", 2),
]


def run_task(job):
    disable, idx, rep = job
    kind, prompt, min_turns = TASKS[idx]
    msgs = [{"role": "user", "content": prompt}]
    choice = {"type": "auto", "disable_parallel_tool_use": disable}
    t0 = time.time()
    turns, per_turn, usage = 0, [], []
    for step in range(10):
        r = ledger.create(cl, "a02", note=f"exp4 disable_parallel={disable}", model=HAIKU, max_tokens=4096,
                          system=SYSTEM, tools=TOOLSET, tool_choice=choice, messages=msgs)
        turns += 1
        usage.append(r.usage.model_dump())
        calls = [b for b in r.content if b.type == "tool_use"]
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        if r.stop_reason != "tool_use":
            break
        per_turn.append([{"name": c.name, "input": c.input} for c in calls])
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": run_tool(c.name, c.input)} for c in calls]})
    secs = time.time() - t0
    final = " ".join(b["text"] for b in msgs[-1]["content"] if b.get("type") == "text")
    rec = {"disable": disable, "idx": idx, "rep": rep, "kind": kind, "min_tool_turns": min_turns, "api_calls": turns,
           "tool_turns": len(per_turn), "tool_calls": sum(len(t) for t in per_turn),
           "calls_per_turn": [len(t) for t in per_turn], "per_turn": per_turn, "secs": round(secs, 2),
           "usage": usage, "final": final[:600], "messages": msgs}
    dump("exp4_parallel.jsonl", rec)
    return rec


if __name__ == "__main__":
    reps = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(TASKS)
    jobs = [(d, i, k) for k in range(reps) for i in range(n) for d in (False, True)]
    random.seed(11)
    random.shuffle(jobs)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for rec in ex.map(run_task, jobs):
            print("seq " if rec["disable"] else "par ", rec["idx"], rec["kind"], rec["calls_per_turn"], rec["api_calls"], rec["secs"])
    print(ledger.report())
