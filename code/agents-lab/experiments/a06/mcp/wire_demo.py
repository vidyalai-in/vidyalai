"""Connect to tickets_server.py through the tap, in modern (auto) and legacy mode, and print the wire."""
import asyncio, json, sys
from pathlib import Path
from mcp import Client
from mcp_types import Implementation
from mcp.client.stdio import StdioServerParameters

HERE = Path(__file__).parent
PY = sys.executable


async def session(mode, log):
    params = StdioServerParameters(command=PY, args=[str(HERE / "tap.py"), str(log), "--", PY, str(HERE / "tickets_server.py")])
    async with Client(params, mode=mode, client_info=Implementation(name="lab-client", version="0.1")) as client:
        print("protocol version:", client.protocol_version)
        tools = await client.list_tools()
        print([t.name for t in tools.tools])
        res = await client.call_tool("search_tickets", {"query": "login"})
        print(res.structured_content)
        res = await client.call_tool("add_comment", {"ticket_id": "T-999", "text": "hi"})
        print(res.is_error, res.content[0].text)
        r = await client.list_resource_templates()
        print([t.uri_template for t in r.resource_templates])
        p = await client.get_prompt("triage", {"ticket_id": "T-101"})
        print(p.messages[0].content.text)


for mode in ("auto", "legacy"):
    log = HERE / f"wire-{mode}.jsonl"
    log.unlink(missing_ok=True)
    asyncio.run(session(mode, log))
