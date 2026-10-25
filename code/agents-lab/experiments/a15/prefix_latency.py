"""Does a cache hit make a call faster? Time one short Haiku 5.5 reply after a long prompt prefix,
with the prefix read from the cache and without caching, at three prefix sizes.

Usage: python prefix_latency.py REPEATS
Each size gets its own random nonce so no other run's cache entry can match. Per size: one call
writes the cache entry, then REPEATS cached and REPEATS uncached calls alternate. Thinking is
disabled and max_tokens is 8, so the call time is mostly prompt processing.
Writes runs/prefix-latency.jsonl.
"""
import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
from multiagent import Limited  # noqa: E402

HERE = Path(__file__).parent
WORLD = json.loads((HERE / "world.json").read_text())["records"]
BLOCK = "\n".join(f"{k}: {v}" for k, v in WORLD.items())  # about 3,000 tokens of records
CL = Limited(ledger.client().with_options(max_retries=6), n=1)
OUT = HERE / "runs" / "prefix-latency.jsonl"


def call(prefix, cached, size, kind):
    system = [{"type": "text", "text": prefix}]
    if cached:
        system[0]["cache_control"] = {"type": "ephemeral"}
    r = ledger.create(CL, "a15", note=f"a15 prefix {size} {kind}", model="claude-haiku-5-5", max_tokens=8,
                      system=system, thinking={"type": "disabled"}, output_config={"effort": "low"},
                      messages=[{"role": "user", "content": "Reply with the single word OK."}])
    u = r.usage
    rec = dict(size=size, kind=kind, seconds=round(CL.local.last, 3), input=u.input_tokens,
               cache_read=u.cache_read_input_tokens or 0, cache_write=u.cache_creation_input_tokens or 0,
               output=u.output_tokens, cost=ledger.cost_of("claude-haiku-5-5", u), ts=time.time())
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


reps = int(sys.argv[1])
for size in (2_000, 20_000, 80_000):
    n = max(1, round(size / 3000))
    prefix = f"Archive copy {secrets.token_hex(8)}. Reference records follow.\n\n" + "\n\n".join(
        f"[copy {i}]\n{BLOCK}" for i in range(n))
    call(prefix, True, size, "write")
    for k in range(reps):
        call(prefix, True, size, "read")
        call(prefix, False, size, "none")
print(ledger.report())
