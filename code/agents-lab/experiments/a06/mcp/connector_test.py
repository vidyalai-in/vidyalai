"""One MCP connector request per model: the API itself is the MCP client (beta mcp-client-2025-11-20)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
from mcp_lab import beta_create

cl = ledger.client()
model = sys.argv[1]
out = open("/home/amit/.cache/vidyalai-scratch/a06/runs/connector.jsonl", "a")
resp = beta_create(
    cl, "a06", note=f"mcp connector {model}", model=model, max_tokens=1500,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{"type": "url", "url": "https://mcp.deepwiki.com/mcp", "name": "deepwiki"}],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "deepwiki"}],
    messages=[{"role": "user", "content": "Use DeepWiki to list the top-level sections of the wiki for "
               "the GitHub repo modelcontextprotocol/python-sdk. Reply with the section titles only."}],
)
d = resp.model_dump(exclude_none=True)
out.write(json.dumps({"model": model, "response": d}) + "\n")
for b in d["content"]:
    print(b["type"], json.dumps({k: v for k, v in b.items() if k != "type"})[:400])
print(resp.stop_reason, resp.usage)
