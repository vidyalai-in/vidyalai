"""Stress catalogs: the 145 hand-written tools plus generated tools from 57 more MCP-style servers
that overlap with them (other trackers, chat apps, mail, file stores, monitoring...).

EXTRA holds 341 generated tools in a fixed shuffled order; CATALOGS[300] and CATALOGS[486] are
the base catalog plus the first 155 or all 341 of them, each listed server by server.
"""
import random

from catalog import CATALOG, T, p, S, I, B

KINDS = {
    "tracker": (["asana", "trello", "clickup", "monday", "youtrack", "shortcut", "zendesk", "freshdesk", "servicenow"], [
        ("search_{items}", "Search {D} {items} by text, status or assignee. Returns ID, title and status.", dict(query=p(S, "Search text"), status=p(S, "Status filter", opt=True))),
        ("get_{item}", "Get one {D} {item} by ID, with description, status and comments.", dict(id=p(S, "{Item} ID"))),
        ("create_{item}", "Create a {D} {item}.", dict(title=p(S, "Title"), description=p(S, "Description", opt=True), project=p(S, "Project or board", opt=True))),
        ("update_{item}", "Update a {D} {item}'s status, priority or title.", dict(id=p(S, "{Item} ID"), status=p(S, "New status", opt=True), priority=p(S, "New priority", opt=True))),
        ("add_comment", "Add a comment to a {D} {item}.", dict(id=p(S, "{Item} ID"), body=p(S, "Comment text"))),
        ("assign_{item}", "Assign a {D} {item} to a user.", dict(id=p(S, "{Item} ID"), assignee=p(S, "Assignee email"))),
        ("list_projects", "List {D} projects or boards.", dict()),
        ("close_{item}", "Close or complete a {D} {item}.", dict(id=p(S, "{Item} ID"))),
    ]),
    "chat": (["discord", "mattermost", "webex", "google_chat", "zoom_chat"], [
        ("post_message", "Post a message to a {D} channel.", dict(channel=p(S, "Channel name or ID"), text=p(S, "Message text"))),
        ("send_direct_message", "Send a direct message to a {D} user.", dict(user=p(S, "User handle or email"), text=p(S, "Message text"))),
        ("search_messages", "Search {D} messages across channels.", dict(query=p(S, "Search query"))),
        ("list_channels", "List {D} channels.", dict()),
        ("get_channel_history", "Get recent messages from a {D} channel.", dict(channel=p(S, "Channel name or ID"), limit=p(I, "Number of messages", opt=True))),
        ("add_reaction", "Add an emoji reaction to a {D} message.", dict(channel=p(S, "Channel"), message_id=p(S, "Message ID"), emoji=p(S, "Emoji name"))),
    ]),
    "mail": (["outlook", "fastmail", "zoho_mail"], [
        ("send_email", "Send an email from the user's {D} account.", dict(to=p("array", "Recipients", items=S), subject=p(S, "Subject"), body=p(S, "Body"))),
        ("search_emails", "Search the user's {D} mailbox.", dict(query=p(S, "Search query"))),
        ("get_email", "Read one {D} message by ID.", dict(message_id=p(S, "Message ID"))),
        ("create_draft", "Create a {D} draft without sending it.", dict(to=p("array", "Recipients", items=S), subject=p(S, "Subject"), body=p(S, "Body"))),
        ("reply_to_email", "Reply to a {D} message.", dict(message_id=p(S, "Message ID"), body=p(S, "Reply body"))),
        ("forward_email", "Forward a {D} message.", dict(message_id=p(S, "Message ID"), to=p("array", "Recipients", items=S))),
        ("archive_email", "Archive a {D} message.", dict(message_id=p(S, "Message ID"))),
    ]),
    "calendar": (["outlook_calendar", "calendly", "zoom"], [
        ("create_event", "Create a {D} event or meeting and invite attendees.", dict(title=p(S, "Title"), start=p(S, "ISO 8601 start"), end=p(S, "ISO 8601 end"), attendees=p("array", "Attendee emails", opt=True, items=S))),
        ("list_events", "List {D} events in a time range.", dict(start=p(S, "Range start"), end=p(S, "Range end"))),
        ("update_event", "Change a {D} event's time or title.", dict(event_id=p(S, "Event ID"), start=p(S, "New start", opt=True), end=p(S, "New end", opt=True))),
        ("delete_event", "Cancel a {D} event.", dict(event_id=p(S, "Event ID"))),
        ("find_free_time", "Find times when people are free in {D}.", dict(attendees=p("array", "Emails", items=S), duration_minutes=p(I, "Length"))),
    ]),
    "files": (["dropbox", "box", "onedrive", "sharepoint", "s3"], [
        ("search_files", "Search {D} files by name or content.", dict(query=p(S, "Search text"))),
        ("read_file", "Read a {D} file's contents.", dict(file_id=p(S, "File ID or path"))),
        ("upload_file", "Upload a file to {D}.", dict(path=p(S, "Destination path"), content=p(S, "File content"))),
        ("share_file", "Share a {D} file with people.", dict(file_id=p(S, "File ID or path"), emails=p("array", "People", items=S))),
        ("list_folder", "List a {D} folder.", dict(folder=p(S, "Folder ID or path"))),
        ("move_file", "Move a {D} file to another folder.", dict(file_id=p(S, "File ID or path"), folder=p(S, "Destination folder"))),
        ("delete_file", "Delete a {D} file.", dict(file_id=p(S, "File ID or path"))),
    ]),
    "docs": (["coda", "quip", "gitbook", "slab", "guru"], [
        ("search", "Search {D} pages by title and content.", dict(query=p(S, "Search text"))),
        ("get_page", "Read a {D} page.", dict(page_id=p(S, "Page ID"))),
        ("create_page", "Create a {D} page.", dict(title=p(S, "Title"), content=p(S, "Content", opt=True))),
        ("update_page", "Update a {D} page's content.", dict(page_id=p(S, "Page ID"), content=p(S, "New content"))),
        ("add_comment", "Comment on a {D} page.", dict(page_id=p(S, "Page ID"), text=p(S, "Comment"))),
    ]),
    "code": (["bitbucket", "azure_devops", "gitea"], [
        ("search_issues", "Search {D} issues.", dict(query=p(S, "Search text"), repo=p(S, "Repository", opt=True))),
        ("create_issue", "Create a {D} issue.", dict(repo=p(S, "Repository"), title=p(S, "Title"))),
        ("list_pull_requests", "List {D} pull requests in a repository.", dict(repo=p(S, "Repository"), state=p(S, "State", opt=True))),
        ("get_pull_request", "Get one {D} pull request.", dict(repo=p(S, "Repository"), number=p(I, "Pull request number"))),
        ("create_pull_request", "Open a {D} pull request.", dict(repo=p(S, "Repository"), title=p(S, "Title"), source=p(S, "Source branch"), target=p(S, "Target branch"))),
        ("merge_pull_request", "Merge a {D} pull request.", dict(repo=p(S, "Repository"), number=p(I, "Pull request number"))),
        ("search_code", "Search code in {D} repositories.", dict(query=p(S, "Code search query"))),
        ("get_file_contents", "Read a file from a {D} repository.", dict(repo=p(S, "Repository"), path=p(S, "Path"))),
        ("list_pipelines", "List {D} CI pipeline runs.", dict(repo=p(S, "Repository"))),
    ]),
    "observability": (["grafana", "newrelic", "splunk", "honeycomb", "prometheus", "elastic"], [
        ("query_metrics", "Query {D} metrics over a time range.", dict(query=p(S, "Metric query"), from_time=p(S, "Range start"))),
        ("search_logs", "Search {D} logs.", dict(query=p(S, "Log query"), from_time=p(S, "Range start"))),
        ("list_alerts", "List {D} alerts and their state.", dict(state=p(S, "State filter", opt=True))),
        ("mute_alert", "Silence a {D} alert.", dict(alert_id=p(S, "Alert ID"), duration_minutes=p(I, "Minutes", opt=True))),
        ("get_dashboard", "Get a {D} dashboard.", dict(dashboard=p(S, "Dashboard name or ID"))),
    ]),
    "incident": (["opsgenie", "incident_io", "firehydrant", "statuspage"], [
        ("list_incidents", "List {D} incidents.", dict(status=p(S, "Status", opt=True))),
        ("acknowledge_incident", "Acknowledge a {D} incident.", dict(incident_id=p(S, "Incident ID"))),
        ("resolve_incident", "Resolve a {D} incident.", dict(incident_id=p(S, "Incident ID"))),
        ("get_oncall", "Get who is on call in {D}.", dict(schedule=p(S, "Schedule name"))),
        ("create_incident", "Open a new {D} incident.", dict(title=p(S, "Title"), service=p(S, "Service", opt=True))),
    ]),
    "db": (["mysql", "snowflake", "bigquery", "mongodb", "redis", "clickhouse"], [
        ("query", "Run a query against {D} and return rows.", dict(query=p(S, "Query"))),
        ("list_tables", "List {D} tables or collections.", dict()),
        ("describe_table", "Describe a {D} table's columns.", dict(table=p(S, "Table name"))),
    ]),
    "billing": (["paddle", "chargebee", "braintree", "recurly"], [
        ("search_customers", "Search {D} customers by email or name.", dict(query=p(S, "Email or name"))),
        ("get_customer", "Get a {D} customer.", dict(customer_id=p(S, "Customer ID"))),
        ("list_charges", "List a {D} customer's payments.", dict(customer_id=p(S, "Customer ID"))),
        ("create_refund", "Refund a {D} payment.", dict(payment_id=p(S, "Payment ID"), amount_cents=p(I, "Amount in cents", opt=True))),
        ("list_invoices", "List a {D} customer's invoices.", dict(customer_id=p(S, "Customer ID"))),
        ("cancel_subscription", "Cancel a {D} subscription.", dict(subscription_id=p(S, "Subscription ID"))),
    ]),
    "crm": (["salesforce", "pipedrive", "zoho_crm", "intercom"], [
        ("search_contacts", "Search {D} contacts.", dict(query=p(S, "Search text"))),
        ("get_contact", "Get a {D} contact.", dict(contact_id=p(S, "Contact ID"))),
        ("create_contact", "Create a {D} contact.", dict(email=p(S, "Email"), name=p(S, "Full name", opt=True))),
        ("list_deals", "List {D} deals or opportunities.", dict(stage=p(S, "Stage", opt=True))),
        ("update_deal", "Update a {D} deal's stage or amount.", dict(deal_id=p(S, "Deal ID"), stage=p(S, "Stage", opt=True))),
        ("log_activity", "Log a call, meeting or note in {D}.", dict(contact_id=p(S, "Contact ID"), type=p(S, "call, meeting or note"), body=p(S, "Notes"))),
    ]),
}
ITEM = {"asana": "task", "trello": "card", "clickup": "task", "monday": "item", "youtrack": "issue", "shortcut": "story",
        "zendesk": "ticket", "freshdesk": "ticket", "servicenow": "incident"}
DISPLAY = {"google_chat": "Google Chat", "zoom_chat": "Zoom Team Chat", "outlook_calendar": "Outlook Calendar",
           "zoho_mail": "Zoho Mail", "azure_devops": "Azure DevOps", "newrelic": "New Relic", "incident_io": "incident.io",
           "zoho_crm": "Zoho CRM", "s3": "Amazon S3", "onedrive": "OneDrive", "sharepoint": "SharePoint", "youtrack": "YouTrack",
           "servicenow": "ServiceNow", "clickup": "ClickUp", "firehydrant": "FireHydrant", "statuspage": "Statuspage",
           "opsgenie": "Opsgenie", "mysql": "MySQL", "bigquery": "BigQuery", "mongodb": "MongoDB", "clickhouse": "ClickHouse",
           "github": "GitHub", "bitbucket": "Bitbucket", "pipedrive": "Pipedrive"}


def _fill(d, server):
    item = ITEM.get(server, "item")
    return {k: (v[0] | {"description": v[0]["description"].replace("{Item}", item.capitalize())}, v[1]) for k, v in d.items()}


EXTRA = []
for kind, (servers, tools) in KINDS.items():
    for s in servers:
        disp = DISPLAY.get(s, s.capitalize())
        item = ITEM.get(s, "item")
        for name, desc, params in tools:
            fmt = dict(D=disp, item=item, items=item + "s")
            EXTRA.append(T(s, name.format(**fmt), desc.format(**fmt), **_fill(params, s)))
random.Random(6).shuffle(EXTRA)
assert len({t["name"] for t in EXTRA + CATALOG}) == len(EXTRA) + len(CATALOG)


def _ordered(tools):  # server by server, as an MCP host lists them
    order = {}
    for t in tools:
        order.setdefault(t["server"], len(order))
    return sorted(tools, key=lambda t: order[t["server"]])


CATALOGS = {145: CATALOG, 300: _ordered(CATALOG + EXTRA[:155]), 486: _ordered(CATALOG + EXTRA)}

if __name__ == "__main__":
    print(len(EXTRA), {k: len(v) for k, v in CATALOGS.items()}, len({t["server"] for t in CATALOGS[486]}), "servers")
