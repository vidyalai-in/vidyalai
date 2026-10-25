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
