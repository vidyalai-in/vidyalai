import asyncio, time
from mcp import Client

async def main():
    t0 = time.perf_counter()
    async with Client("https://mcp.deepwiki.com/mcp") as c:
        print("negotiated", c.protocol_version, "in", round(time.perf_counter() - t0, 2), "s")
        tools = (await c.list_tools()).tools
        print(len(tools), [t.name for t in tools])

asyncio.run(main())
