import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from mcp_lab import MCPServers, stdio

with MCPServers({"tickets": stdio("/home/amit/.cache/vidyalai-scratch/a06/mcp/tickets_server.py")}) as s:
    for t in s.tools:
        print(json.dumps(t.schema()))
    print(s.tools[0].fn(query="csv"))
    try:
        s.tools[1].fn(ticket_id="T-9", text="x")
    except Exception as e:
        print("error:", e)
