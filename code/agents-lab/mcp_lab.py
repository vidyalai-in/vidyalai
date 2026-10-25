"""Part 6: MCP servers as tools for the lab harness.

An MCP client (the official `mcp` Python SDK) connects to each server, lists its tools and turns
every MCP tool into a harness Tool: the server's inputSchema becomes the API's input_schema, the
name gets a server prefix, and calling the tool sends `tools/call` to the server. The agent loop
itself (react.run_loop) is unchanged.

The MCP client is async and the lab loop is sync, so the client lives in an anyio blocking portal
(an event loop on a helper thread) and each tool call crosses into it with portal.call().
"""

import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path

from anyio.from_thread import start_blocking_portal
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import Implementation

sys.path.insert(0, str(Path(__file__).parent))
from harness import Tool  # noqa: E402


class MCPTool(Tool):
    """A harness Tool whose schema comes verbatim from an MCP server's tools/list."""

    def __init__(self, server, mcp_tool, call):
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}__{mcp_tool.name}")[:128]  # API name rule
        super().__init__(name, mcp_tool.description or "", {}, call)
        self.input_schema = mcp_tool.input_schema

    def schema(self):
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _as_text(result):
    """CallToolResult -> the string the model sees; MCP execution errors become exceptions,
    which the harness returns to the model as is_error tool results."""
    text = "\n".join(c.text if c.type == "text" else json.dumps(c.model_dump()) for c in result.content)
    if result.is_error:
        raise RuntimeError(text)
    return text


class MCPServers:
    """with MCPServers({"tickets": StdioServerParameters(...)}) as servers: tools = servers.tools"""

    def __init__(self, servers, mode="auto"):
        self.servers, self.mode, self.tools = servers, mode, []

    def __enter__(self):
        self._stack = ExitStack()
        portal = self._stack.enter_context(start_blocking_portal())
        for name, params in self.servers.items():
            client = self._stack.enter_context(portal.wrap_async_context_manager(
                Client(params, mode=self.mode, client_info=Implementation(name="vidyalai-lab", version="0.6"))))
            for t in portal.call(client.list_tools).tools:
                def call(_client=client, _name=t.name, **args):
                    return _as_text(portal.call(_client.call_tool, _name, args))
                self.tools.append(MCPTool(name, t, call))
        return self

    def __exit__(self, *exc):
        self._stack.close()


def stdio(script, *args):
    """StdioServerParameters that run a Python MCP server script with this interpreter."""
    return StdioServerParameters(command=sys.executable, args=[str(script), *args])


# ---- Beta endpoints (MCP connector) with the same budget accounting as ledger.create -----------

def beta_create(cl, part, note="", est_input_tokens=20_000, **kwargs):
    """client.beta.messages.create with ledger.guard() before and ledger.record() after.
    count_tokens can't see the remote server's tools, so the caller passes an input estimate."""
    import ledger
    model = kwargs["model"]
    if model not in ledger.ALLOWED_MODELS:
        raise ValueError(f"Model {model} not allowed")
    r = ledger._rates(model, est_input_tokens)
    ledger.guard(part, (est_input_tokens * r["cache_write"] + kwargs.get("max_tokens", 4096) * r["out"]) / 1e6)
    resp = cl.beta.messages.create(**kwargs)
    ledger.record(part, model, resp.usage, note)
    return resp
