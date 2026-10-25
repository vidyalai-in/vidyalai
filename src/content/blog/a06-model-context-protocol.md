---
title: "MCP: Connecting Agents to Tools"
description: "How the Model Context Protocol works on the wire, how to plug MCP servers into an agent loop, and what happens to tool choice as tools pile up."
slug: "model-context-protocol"
category: "agents-tool-use"
tags: ["agents", "tool-use", "mcp", "caching", "cost"]
series: "agentic-ai"
seriesOrder: 6
pubDate: 2026-10-24
---

## TL;DR

The Model Context Protocol (MCP) lets an AI application discover and call tools in separate programs called servers, using JSON-RPC 2.0 messages over stdio or HTTP. Its current revision, 2026-07-28, dropped the opening handshake: every request now carries its own version and capabilities. This part builds a server with the official Python SDK, captures a real session, plugs MCP tools into the lab harness, and measures tool choice as tools pile up. On 79 requests with a known right answer, Claude Haiku 5.5 chose the right tool and arguments every time with 145 tools loaded and 99.4% of the time with 486, while the prompt grew to 55,743 tokens. Tool search cut the prompt to about 2,900 tokens but added 1.8 seconds and a new failure: the model searches for what it expects, and misses the right tool when it expects the wrong one.

## Why it matters

Connect 18 MCP servers to an agent, as the experiment below does, and it has 145 tools. Their definitions cost 19,529 input tokens on every request to Claude Haiku 5.5, before the user types a word. At 486 tools it's 55,753, about what Anthropic's docs quote for a typical GitHub, Slack, Sentry, Grafana and Splunk setup.

Anthropic's [tool search docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) warn that Claude picks tools less accurately once it has more than 30 to 50 of them, and offer a fix: mark definitions with `defer_loading` and let the model search for the ones it needs. This part tests both claims on a task set where the right tool is known.

MCP is why so many tools arrive at once: a server written once works in Claude, ChatGPT, Cursor, Gemini, Microsoft Copilot and VS Code, the adopters Anthropic listed when it [donated MCP](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) to the Linux Foundation in December 2025. This part builds on [Part 2](/blog/tool-calling-internals/), which showed how tool definitions become prompt tokens, and on [Building Tool-Using Agents That Work Reliably](/blog/reliable-tool-using-agents/) (System Design 10).

## The intuition

Think of a wall socket. Before standard plugs, every appliance came with its own wiring. With one plug shape, any appliance works in any socket. MCP is the plug: a server (the appliance) exposes tools in one format, and any host application (the house) with an MCP client (the socket) can use them.

The analogy breaks in two places. A plug says nothing about what the appliance does, but an MCP server describes each tool in words the model reads, and those words steer it, for better or worse. And an idle appliance costs nothing, while every tool definition an agent holds is paid for in tokens on every request.

## Step 1: the protocol on the wire

### Hosts, clients, servers and JSON-RPC

The [specification](https://modelcontextprotocol.io/specification/2026-07-28) names three roles. A host is the AI application, such as a chat app, an IDE or the lab harness from [Part 1](/blog/what-is-an-ai-agent/). A client is the connector inside the host that talks to one server. A server is a program that offers tools, data or prompt templates.

Every message is JSON-RPC 2.0, a small standard for remote procedure calls written as JSON. A request has a `method`, `params` and an `id` that must not be null. A response repeats the `id` and carries either a `result` or an `error` with an integer `code` and a `message`. A notification has no `id` and gets no reply.

### What changed in 2026-07-28

The [current revision](https://modelcontextprotocol.io/specification/2026-07-28/changelog) made MCP stateless. Earlier revisions opened each connection with an `initialize` handshake that swapped versions and capabilities. That is gone, along with HTTP sessions and `ping`. Every request now carries the protocol version and the client's capabilities in a `_meta` field, and a server must not rely on earlier requests. State that spans calls, such as a shopping cart, becomes a handle the model passes as an ordinary tool argument.

Capability negotiation now happens in two halves. The server describes itself through `server/discover`, which every server must implement: its supported versions, its identity, and its capabilities, meaning which features it offers (tools, resources, prompts) and whether it announces changes to them. The client declares its own capabilities on every request, and a server that needs one the request didn't declare must fail with error `-32021`. A request for a version the server doesn't speak gets `-32022` and the list it does.

### A real session

The [official Python SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp` 2.3.0, released October 2, 2026) speaks both styles. A 30-line tap script sat between the SDK's client and the ticket server from Step 4, logging every line. In the client's default `auto` mode the session opened like this (one message per line, as on the wire):

```text
client→server {"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"lab-client","version":"0.1"},"io.modelcontextprotocol/clientCapabilities":{}}}}
server→client {"jsonrpc":"2.0","id":1,"result":{"cacheScope":"private","capabilities":{"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true},"tools":{"listChanged":true}},"instructions":"Ticket tracker for the Acme web app.","resultType":"complete","supportedVersions":["2026-07-28"],"ttlMs":0,"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"acme-tickets","version":"1.0.0"}}}}
```

Next came `tools/list`, which returned each tool's name, description and `inputSchema` (a JSON Schema for its arguments), then a call:

```text
client→server {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_tickets","arguments":{"query":"login"},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"lab-client","version":"0.1"},"io.modelcontextprotocol/clientCapabilities":{}}}}
server→client {"jsonrpc":"2.0","id":3,"result":{"content":[{"text":"{\n  \"id\": \"T-101\",\n  \"title\": \"Login times out on mobile\",\n  \"status\": \"open\",\n  \"comments\": []\n}","type":"text"}],"isError":false,"resultType":"complete","structuredContent":{"result":[{"id":"T-101","title":"Login times out on mobile","status":"open","comments":[]}]},"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"acme-tickets","version":"1.0.0"}}}}
```

The answer arrives twice, as `content` for the model to read and `structuredContent` for code. List results also carry `ttlMs` and `cacheScope`, hints for how long a client may cache them. With `mode="legacy"`, the same session opens the 2025-11-25 way:

```text
client→server {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"lab-client","version":"0.1"}}}
server→client {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"prompts":{"listChanged":false},"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":false}},"instructions":"Ticket tracker for the Acme web app.","protocolVersion":"2025-11-25","serverInfo":{"name":"acme-tickets","version":"1.0.0"}}}
client→server {"jsonrpc":"2.0","method":"notifications/initialized"}
client→server {"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

Statelessness costs bytes: the same `tools/call` took 299 bytes in the new style and 111 in the old. The old style is still common. When the SDK's client connected to DeepWiki's public server (`https://mcp.deepwiki.com/mcp`) on October 25, 2026, the `server/discover` probe failed and the client fell back to `initialize` at 2025-11-25, as the [stdio spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio#backward-compatibility) tells clients to do.

### Two kinds of error

A comment on a ticket that doesn't exist came back as a normal result:

```text
server→client {"jsonrpc":"2.0","id":4,"result":{"content":[{"text":"Error executing tool add_comment: No ticket T-999","type":"text"}],"isError":true,"resultType":"complete","_meta":{"io.modelcontextprotocol/serverInfo":{"name":"acme-tickets","version":"1.0.0"}}}}
```

The [tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools#error-handling) separates protocol errors (an unknown tool, a malformed request), sent as JSON-RPC errors, from execution errors like this one, flagged `isError: true` so the model can read them and retry. In the SDK, the message reaches the model only if the tool raises `ToolError`.

## Step 2: what a server can offer

![A sequence diagram with four participants: the lab loop (the host), the MCP client, the MCP server over stdio, and the Claude API. The client sends server/discover to the server and gets back versions and capabilities. It sends tools/list and gets back names, descriptions and input schemas, which it hands to the lab loop with tool names renamed to server__tool. The lab loop sends messages and tool definitions to the Claude API and gets back a tool_use block for tickets__search_tickets. The loop asks the client to call search_tickets; the client sends tools/call with _meta to the server, gets back content with isError false, and returns the result text to the loop. The loop sends a tool_result to the Claude API, which returns the next tool_use or a final answer.](/diagrams/a06-mcp-session.svg "One MCP-backed tool call in the lab harness.")

Only tool definitions and tool results reach the model; the host's code makes every MCP call.

### Three primitives

Servers offer three kinds of things, and the spec gives each a different controller:

| Primitive | What it is | Who decides to use it | Methods |
|---|---|---|---|
| Tools | Functions with a JSON Schema for their arguments | The model ("model-controlled") | `tools/list`, `tools/call` |
| Resources | Data identified by a URI, such as a file or a database row | The host application ("application-driven") | `resources/list`, `resources/read`, `resources/templates/list` |
| Prompts | Message templates with arguments, often shown as slash commands | The user ("user-controlled") | `prompts/list`, `prompts/get` |

Only tools reach the model directly. The ticket server in Step 4 also has a resource template, `tickets://{ticket_id}`, and a prompt, `triage`; the host decides whether to paste the one into context or offer the other in a menu.

### When the server needs the client

Sometimes a server needs something only the client has. Elicitation asks the user for input through a form, or sends them to a URL for anything sensitive (servers must not ask for passwords or API keys through a form). Sampling asks the client's model for a completion, and roots lists the user's folders. Since 2026-07-28 these travel as multi round-trip requests: the server answers with `resultType: "input_required"`, and the client gathers the answers and retries the original request with them attached. The same revision [deprecated](https://modelcontextprotocol.io/specification/2026-07-28/deprecated) sampling, roots and server logging, which become eligible for removal after July 2027.

### Security

Tool descriptions go straight into the prompt, so a malicious server can inject instructions before the user types anything. Invariant Labs named this a [tool poisoning attack](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks) in April 2025, along with the "rug pull," where a server changes a description after the user approved it. The spec says annotations "should be considered untrusted, unless obtained from a trusted server" and that hosts "must obtain explicit user consent before invoking any tool." [System Design 17](/blog/prompt-injection-defense-tool-agents/) covers the defenses; Part 13 covers sandboxing.

## Step 3: transports and authorization

With the stdio transport, the client launches the server as a subprocess and they exchange newline-delimited JSON-RPC over its standard input and output. The server writes nothing else to stdout (logs go to stderr), and the client shuts it down by closing stdin, escalating to `SIGTERM` and `SIGKILL` if needed. On this machine (WSL2, Python 3.12, `mcp` 2.3.0), starting the server took about 470 ms, and 1,000 `tools/call` round trips had a median of 1.4 ms (95th percentile under 2 ms) in both styles. Next to a one-second model call, the protocol adds almost nothing.

Remote servers use Streamable HTTP: one endpoint that accepts a POST per message and answers with a JSON object or an SSE (server-sent events) stream carrying progress notifications and then the response. Every POST carries an `MCP-Protocol-Version` header, and since 2026-07-28 also `Mcp-Method` and, for calls, `Mcp-Name`, so gateways can route on headers. Servers must validate the `Origin` header to block DNS rebinding (a web page tricking a browser into calling a server on localhost), and local servers should bind to 127.0.0.1.

[Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) is optional and HTTP-only; a stdio server takes credentials from its environment. A protected server acts as an OAuth 2.1 resource server. A request without a token gets HTTP 401 and a pointer to the server's protected resource metadata (RFC 9728), which names its authorization server. The client registers, preferably with a Client ID Metadata Document (an HTTPS URL that serves as its client ID), sends the user through a browser login with PKCE and a `resource` parameter naming the server, and attaches the access token to every request.

## Step 4: MCP servers in the lab loop

### The server

Here is the whole ticket server, built with the SDK's `MCPServer`. Type hints become the `inputSchema`, docstrings become descriptions, and `mcp.run()` serves stdio:

```python
"""acme-tickets: a small MCP server over an in-memory ticket tracker (stdio transport)."""
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

TICKETS = {
    "T-101": {"title": "Login times out on mobile", "status": "open", "comments": []},
    "T-102": {"title": "CSV export drops the last row", "status": "open", "comments": []},
    "T-103": {"title": "Dark mode colors on billing page", "status": "closed", "comments": []},
}

mcp = MCPServer("acme-tickets", version="1.0.0", instructions="Ticket tracker for the Acme web app.")


@mcp.tool()
def search_tickets(query: str, status: str = "open") -> list[dict]:
    """Search tickets whose title contains the query (case-insensitive). status: open, closed or any."""
    return [{"id": k, **v} for k, v in TICKETS.items()
            if query.lower() in v["title"].lower() and status in ("any", v["status"])]


@mcp.tool()
def add_comment(ticket_id: str, text: str) -> str:
    """Add a comment to a ticket, by its ID (for example T-101)."""
    if ticket_id not in TICKETS:
        raise ToolError(f"No ticket {ticket_id}")
    TICKETS[ticket_id]["comments"].append(text)
    return f"Comment added to {ticket_id} ({len(TICKETS[ticket_id]['comments'])} total)"


@mcp.resource("tickets://{ticket_id}")
def ticket(ticket_id: str) -> dict:
    """One ticket as JSON."""
    return TICKETS[ticket_id]


@mcp.prompt()
def triage(ticket_id: str) -> str:
    """Ask the model to triage one ticket."""
    return f"Read ticket {ticket_id}, decide its severity (P1-P3) and suggest an owner."


if __name__ == "__main__":
    mcp.run()  # stdio: JSON-RPC on stdin/stdout, logs on stderr
```

### Mapping MCP tools to Claude tools

The Part 1 harness takes `Tool` objects with a name, a description, a schema and a Python function. The bridge makes one per MCP tool: the schema passes through untouched, the name gets a server prefix (tool names are unique only within a server, and the spec recommends prefixing when aggregating), and the function sends `tools/call`. The Claude API allows names matching `^[a-zA-Z0-9_-]{1,128}$` while MCP also allows dots, hence the substitution. The SDK client is asynchronous and the harness isn't, so the clients live in an anyio blocking portal, an event loop on a helper thread:

```python
# lab/mcp_lab.py
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
```

Nothing in the agent loop changed. This runs Part 3's `run_loop` against the server:

```python
"""Part 6: the lab loop (react.run_loop) driving an MCP server over stdio, on Claude Haiku 5.5."""
from pathlib import Path

from mcp_lab import MCPServers, stdio
from react import run_loop

SERVER = Path(__file__).with_name("tickets_server.py")
task = "Find the open ticket about CSV exports and add a comment to it: 'Fix ships in 4.12'."

with MCPServers({"tickets": stdio(SERVER)}) as servers:
    result = run_loop(task, servers.tools, "You manage Acme's ticket tracker.", part="a06",
                      model="claude-haiku-5-5", effort="medium", max_steps=6, verbose=True,
                      transcript="mcp_demo.jsonl")
print(result.stop, f"${result.cost:.5f}")
print(result.answer)
```

```text
step  1 out    77 (thinking     0)  0.94s $0.00005 tickets__search_tickets({"query": "CSV export", "status": "open"})
step  2 out    88 (thinking     0)  0.88s $0.00005 tickets__add_comment({"ticket_id": "T-102", "text": "Fix ships in 4.12"})
step  3 out    54 (thinking     0)  0.88s $0.00004 end_turn
end_turn $0.00013
I added the comment "Fix ships in 4.12" to ticket T-102, "CSV export drops the last row." It was the only open ticket matching "CSV export."
```

Three model calls on `claude-haiku-5-5` (effort `medium`, adaptive thinking, which it didn't use), two MCP round trips, $0.00013. The first call hit a 623-token cache entry written by an identical run minutes earlier, which had cost $0.00023.

### When the API is the MCP client

The Claude API can also be the client. The [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) (beta header `mcp-client-2025-11-20`) takes an `mcp_servers` list of HTTPS URLs plus an `mcp_toolset` entry per server in `tools`, where you can allowlist, denylist or defer tools. Anthropic's servers list the tools and call them, and the response contains `mcp_tool_use` and `mcp_tool_result` blocks, with no loop in your code. Only tools are supported, the server must be on public HTTPS (no local stdio), you run the OAuth flow yourself, and the feature isn't covered by zero data retention. A newer header, `mcp-client-2026-09-15`, pins a server's tool list for the conversation.

The docs don't list model restrictions, so one request per model settled it. Pointed at DeepWiki and asked for a repository's wiki sections, both `claude-haiku-5-5` and `claude-sonnet-5-5` called `read_wiki_structure` and answered, with 2,611 input tokens each. For local servers, resources or prompts, use your own client.

## Step 5: tool search and deferred loading

[Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) is the API's answer to too many tools. You still send every definition, mark most with `"defer_loading": true`, and add a search tool: `tool_search_tool_regex_20251119`, where the model writes Python regular expressions, or `tool_search_tool_bm25_20251119`, where it writes a plain-language query. BM25 is the classic search-engine ranking formula: a tool scores higher the more often its name, description and arguments contain the query's words, with rare words weighted more.

The search runs on Anthropic's servers and returns five `tool_reference` blocks by default (the model may ask for more), which the API expands into full definitions inline. Deferred tools stay out of the system-prompt prefix, so discovering a tool doesn't invalidate the cache. A request may defer up to 10,000 tools, one must stay loaded, and searches aren't billed separately; the definitions they load count as input tokens. Both variants are listed for Haiku 5.5 and Sonnet 5.5 with no beta header, and the connector takes `defer_loading` per server.

The free [token-counting endpoint](https://platform.claude.com/docs/en/build-with-claude/token-counting) shows the effect on one request with the system prompt used below:

| Tools available | All loaded (input tokens) | Tool search, all deferred (input tokens) |
|---|---|---|
| 5 | 1,335 | 1,056 |
| 20 | 3,468 | 1,056 |
| 50 | 7,412 | 1,056 |
| 100 | 13,758 | 1,056 |
| 145 | 19,529 | 1,056 |
| 300 | 35,820 | 1,085 |
| 486 | 55,753 | 1,085 |

All-loaded grows by about 130 tokens per tool. The deferred prompt stays flat until a search pulls definitions in. (The last two rows use a system prompt one sentence longer, explained below.)

## Step 6: does the model get lost in a big toolbox?

### The setup

The catalog has 145 tools from 18 servers, written like real MCP tools, with deliberate near-duplicates: five `search_issues` tools (GitHub, GitLab, Linear, Jira, Sentry), three ways to send a message (Slack, Teams, Gmail), meeting tools in both Google Calendar and Teams, and search in Drive, Notion, Confluence and the local filesystem.

There are 79 requests, one per target tool, each with the values the call must contain: "Refund $25 of charge ch_3PqR." must call `stripe__create_refund` with `charge_id` `ch_3PqR` and `amount_cents` 2500. The system prompt says where things live at the fictional company (engineering work in Linear, support tickets in Jira, partners on Teams) and asks for one tool call. A request with N tools gets its target plus N−1 others from a shuffle seeded per request, so each smaller set sits inside the larger ones.

Every run used `claude-haiku-5-5` at effort `medium` with adaptive thinking. Tools were offered three ways: all loaded; deferred behind the BM25 or regex search tool; or filtered by my own client-side BM25, which loads the top 5 matches and needs no extra model call. A trial counts as correct when the first tool call names the target with every required value. Each request ran 3 times (2 for regex and client-side retrieval). Rates carry 95% Wilson intervals, a confidence interval for proportions that behaves well near 100%.

### The results

| Tools | How offered | Trials | Right tool and arguments | Prompt tokens per request | Cost per 1,000 requests | Median latency |
|---|---|---|---|---|---|---|
| 5 | all loaded | 237 | 100% (98.4-100) | 1,339 | $0.18 | 1.63 s |
| 20 | all loaded | 237 | 100% (98.4-100) | 3,284 | $0.37 | 1.66 s |
| 50 | all loaded | 237 | 100% (98.4-100) | 7,167 | $0.76 | 1.73 s |
| 100 | all loaded | 237 | 100% (98.4-100) | 13,683 | $1.42 | 1.84 s |
| 145 | all loaded, cached | 237 | 100% (98.4-100) | 19,519 | $0.29 | 1.79 s |
| 20 | tool search, BM25 | 237 | 100% (98.4-100) | 2,821 | $0.22 | 3.61 s |
| 50 | tool search, BM25 | 237 | 98.7% (96.3-99.6) | 2,875 | $0.23 | 3.49 s |
| 100 | tool search, BM25 | 237 | 98.7% (96.3-99.6) | 2,893 | $0.23 | 3.55 s |
| 145 | tool search, BM25 | 237 | 98.7% (96.3-99.6) | 2,893 | $0.23 | 3.57 s |
| 145 | tool search, regex | 158 | 93.0% (88.0-96.1) | 3,751 | $0.33 | 3.78 s |
| 20 | client BM25, top 5 | 158 | 92.4% (87.2-95.6) | 1,397 | $0.20 | 1.94 s |
| 145 | client BM25, top 5 | 158 | 87.3% (81.3-91.7) | 1,374 | $0.20 | 1.98 s |

With every tool loaded, Haiku 5.5 was right in all 1,185 trials from 5 to 145 tools. To push harder, a stress run added 341 generated tools from 57 more servers (Outlook, Dropbox, Asana, Zendesk, Grafana, Snowflake and others, all overlapping the originals) and one sentence to the system prompt: other servers belong to other teams, so use them only when a request names them. Each request ran twice:

| Tools | How offered | Trials | Right tool and arguments | Prompt tokens per request | Cost per 1,000 requests | Median latency |
|---|---|---|---|---|---|---|
| 145 | all loaded, cached | 158 | 100% (97.6-100) | 19,548 | $0.31 | 2.04 s |
| 300 | all loaded, cached | 158 | 100% (97.6-100) | 35,810 | $0.53 | 2.23 s |
| 486 | all loaded, cached | 158 | 99.4% (96.5-99.9) | 55,743 | $0.79 | 2.64 s |
| 145 | tool search, BM25 | 158 | 98.7% (95.5-99.7) | 2,954 | $0.24 | 3.68 s |
| 300 | tool search, BM25 | 158 | 98.7% (95.5-99.7) | 2,949 | $0.23 | 3.71 s |
| 486 | tool search, BM25 | 158 | 98.7% (95.5-99.7) | 2,957 | $0.24 | 3.89 s |

At 486 tools Haiku made one mistake in 158 trials, and the all-loaded intervals all overlap: no measurable decline. The docs' 30-to-50 warning didn't show up on this task set, which had clear requests, one- or two-sentence descriptions and a system prompt that named each system's job. Descriptions matter more: in [Part 2](/blog/tool-calling-internals/), terse ones cost 12 points with only ten tools.

![Line chart of prompt tokens per request against the number of tools available, from 5 to 486. With all tools loaded, tokens rise in a straight line from 1,339 at 5 tools through 13,683 at 100 and 19,519 at 145 to 55,743 at 486. With BM25 tool search and every tool deferred, tokens stay flat between 2,821 at 20 tools and 2,957 at 486.](/diagrams/a06-tool-tokens.svg "Mean prompt tokens per request on claude-haiku-5-5 (input plus cache reads and writes), 79 requests.")

Tool search cut prompt tokens by 85% at 145 tools and 95% at 486, in line with the docs' "over 85 percent."

### Where tool search went wrong

Every BM25 error was one request: "Set up a meeting 'Partner sync' with dana@partner.com on 2026-10-29 from 16:00 to 16:30 UTC." With all tools loaded, Haiku saw `teams__create_meeting`, remembered that partners use Teams, and called it. With search, it chose what to look for before seeing any options. Its thinking, in full: "I need to schedule this meeting on Google Calendar, so I should search for the right tool to create a calendar event." It searched `create Google Calendar event meeting`, got five Google Calendar tools, and booked a calendar event; the Teams tool never entered its context. (The single all-loaded mistake at 486 tools was this request too.)

The regex variant failed 11 times in 158 for another reason: in these runs it returned matches in catalog order, five at most. Asked to create a private Slack channel, Haiku searched `slack.*channel|channel.*slack|create.*channel`, which matched every Slack tool that mentions a channel, and the first five didn't include `slack__create_channel`. It answered, "I can't create the channel. None of the Slack tools I have can create channels." Seven of the 11 failures were refusals like this. Two sentences added to the system prompt (searches return at most 5 tools in catalog order; if none does exactly what was asked, search again more narrowly or with a higher limit) raised regex accuracy to 98.1% (155 of 158, CI 94.6-99.4). Per request, 8 improved and 1 got worse, a sign-test p of about 0.04.

Client-side retrieval failed only at retrieval. When the target was among the 5 tools loaded, Haiku was right 568 times out of 568. My BM25 ranked the target in the top 5 for 73 of 79 requests at 20 tools and 69 of 79 at 145, and accuracy fell with it. "Find the product spec for offline mode" retrieved Google Calendar's `find_free_time`, a filesystem search and three GitHub tools, so Haiku searched the local disk for `**/*offline*`. It was the cheapest arm, but nothing recovers a tool the retriever missed.

### Tokens, money and time

Fewer tokens didn't mean much less money, because of prompt caching. When the tool list is identical across requests, the definitions are read from cache at a tenth of Haiku 5.5's input price: 145 cached tools cost $0.29 per 1,000 requests against $0.23 with search, and 486 cost $0.79 against $0.24. Without caching, 100 tools cost $1.42. A host that adds and drops servers per request, or a server that reorders its tools, pays full price.

Search costs time. The API runs the model twice in one request, once to search and once to call the tool, so output roughly doubled (about 220 tokens against 100) and median latency rose by 1.8 seconds at 145 tools. Where search pays off is context space: at 486 tools, all-loaded used 55,743 tokens of the window before the conversation began.

### A stronger model

`claude-sonnet-5-5` ran the regex variant once on every other request (40 requests, effort `medium`, adaptive thinking). It got 34 right, 85.0% (CI 70.9-92.9), against Haiku's 73 of 80, 91.2% (CI 83.0-95.7), on the same 40: within noise, at $6.40 per 1,000 requests against $0.36. It failed the same way. For "Remind me to renew the SSL certificate," it searched `remind|calendar|event|reminder`, got the first five Google Calendar tools, and created an event, because `gcal__create_reminder` is eighth in that server's list.

The experiment's 4,070 trials cost $1.78; with pilots, demos and the connector calls, this part spent $1.81.

## In the real world

- Anthropic's [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) post (November 24, 2025) reports tool definitions reaching 134K tokens at Anthropic before optimization, and on internal MCP evaluations "Opus 4 improved from 49% to 74%, and Opus 4.5 improved from 79.5% to 88.1% with Tool Search Tool enabled." Those are older models on different tasks, not measured here.
- The MCP project's [client best practices](https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices) suggest switching to on-demand tool discovery once definitions pass 1% to 5% of the context window.
- The 2026-07-28 spec asks servers to return `tools/list` in a deterministic order "to enable client-side caching and improve LLM prompt cache hit rates," the effect the cost table shows.
- Anthropic counted "more than 10,000 active public MCP servers" and "97M+ monthly SDK downloads across Python and TypeScript" when it [donated the protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) on December 9, 2025, a year after [announcing it](https://www.anthropic.com/news/model-context-protocol).

## Common misconceptions

- **"More tools always means worse tool choice."** Not here: Haiku 5.5 was right in every all-loaded trial up to 145 tools and 157 of 158 at 486. Clear descriptions and a system prompt that maps systems to jobs carried it.
- **"Tool search is free accuracy."** It cut tokens by 85% to 95% and was slightly less accurate (98.7% against 100% at 145 tools), because the model picks its search terms before it sees the options.
- **"Fewer tokens means a cheaper agent."** Only without caching. A stable, cached list of 145 tools cost $0.29 per 1,000 requests against $0.23 with search, and search was 1.8 seconds slower.
- **"MCP sessions start with initialize."** Not since 2026-07-28, when per-request `_meta` and `server/discover` replaced the handshake; clients still fall back to `initialize` for older servers.
- **"An MCP server is a trusted plugin."** Its descriptions are prompt text written by someone else, and the spec tells hosts to treat annotations as untrusted.

## Key takeaways

- MCP is JSON-RPC 2.0 between a host's client and a server, over stdio or Streamable HTTP. Since 2026-07-28 every request stands alone, which costs bytes but no measurable time: a stdio round trip took 1.4 ms.
- Servers offer tools for the model, resources for the app and prompts for the user. Turning MCP tools into Claude tools is a rename plus the server's own JSON Schema; the agent loop doesn't change.
- Tool count alone didn't hurt Haiku 5.5, up to 486 tools. Tokens grew in a straight line, to 55,743 per request.
- Tool search keeps the prompt near 2,900 tokens at any size, at the price of a second model pass, 1.8 seconds, and misses when the model searches for the wrong thing. Telling it how search works lifted regex search from 93.0% to 98.1%.
- Keep tool lists stable so caching works, and treat every server's descriptions as untrusted input.

## What's next

So far there has been one agent with many tools. Part 7 splits the work across agents: an orchestrator that hands subtasks to workers, sub-agents with their own context windows, and agent-to-agent protocols, measured as accuracy and token multiples against a single agent on the same tasks.

## Further reading

- [Model Context Protocol specification, revision 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28): the normative spec used throughout; start with the [key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog) since 2025-11-25.
- [MCP transports: stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio) and [Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http): framing, headers, shutdown and backward compatibility.
- [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization): the OAuth 2.1 flow for HTTP servers.
- [MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools): definitions, naming, errors, state handles and security requirements.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk): the server and client used here ([docs](https://py.sdk.modelcontextprotocol.io/)).
- [JSON-RPC 2.0 specification](https://www.jsonrpc.org/specification): the message format underneath.
- [Tool search tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool): deferred loading, both search variants, limits and model support.
- [MCP connector (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector): remote MCP servers from the Messages API, toolset configuration and pinning.
- [Advanced tool use (Bin Wu, Anthropic, 2025)](https://www.anthropic.com/engineering/advanced-tool-use): why tool search exists, with Anthropic's token and accuracy numbers.
- [MCP client best practices](https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices): progressive discovery and programmatic tool calling for hosts with many servers.
- [MCP security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices): confused deputies, token passthrough, SSRF and local server compromise.
- [MCP Security Notification: Tool Poisoning Attacks (Beurer-Kellner and Fischer, Invariant Labs, 2025)](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks): hidden instructions in tool descriptions, and rug pulls.
- [Introducing the Model Context Protocol (Anthropic, 2024)](https://www.anthropic.com/news/model-context-protocol): the original announcement.
