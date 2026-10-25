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
