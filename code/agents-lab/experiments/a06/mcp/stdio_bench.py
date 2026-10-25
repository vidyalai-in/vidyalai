"""Round-trip time of tools/call over stdio (no tap), modern vs legacy mode, 1,000 calls each."""
import asyncio, statistics, sys, time
from pathlib import Path
from mcp import Client
from mcp.client.stdio import StdioServerParameters

SERVER = str(Path(__file__).with_name("tickets_server.py"))


async def bench(mode, n=1000):
    t0 = time.perf_counter()
    async with Client(StdioServerParameters(command=sys.executable, args=[SERVER]), mode=mode) as c:
        connect = time.perf_counter() - t0
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            await c.call_tool("search_tickets", {"query": "login"})
            ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{mode:6s} connect {connect*1000:.0f} ms  call median {statistics.median(ts)*1e3:.2f} ms  p95 {ts[int(.95*n)]*1e3:.2f} ms")

for mode in ("auto", "legacy", "auto", "legacy"):
    asyncio.run(bench(mode))
