"""74 machine-checkable requests. Each names the one correct tool (sometimes an accepted alternative)
and the argument values that must appear in the call. Unlisted optional arguments are not checked."""

SYSTEM = """You are the operations assistant for Acme, a software company. Where things live at Acme:
- Code: GitHub (org acme) for the web app and backend; GitLab (group acme-mobile) for the iOS and Android apps.
- Engineering work: Linear (team keys ENG, WEB, MOB; issue IDs like ENG-123). Customer support tickets: Jira project SUP (keys like SUP-123).
- Production errors: Sentry. Metrics, logs and monitors: Datadog. Paging and on-call: PagerDuty.
- Product specs and meeting notes: Notion. Engineering runbooks and postmortems: Confluence (space OPS). Shared files: Google Drive.
- Chat: Slack. The sales team and outside partners use Microsoft Teams. Email: Gmail. Calendar: Google Calendar.
- Billing: Stripe. CRM: HubSpot. Analytics database: Postgres. Local files: the workspace filesystem.
Fulfil each request with one tool call: choose the single tool that does what is asked and call it with the values the request gives. Do not ask questions."""

# matcher forms: ("eq", v) normalized equality; ("has", s) substring; ("any", [s, ...]) any substring; ("set", [...]) same set
R = []


def req(id_, text, tool, alt=None, /, **args):
    R.append({"id": id_, "text": text, "tool": tool, "args": args, "alt": alt or {}})


eq = lambda v: ("eq", v)
has = lambda s: ("has", s)
st = lambda *v: ("set", list(v))

req("gh1", "Open a GitHub issue in acme/api titled 'Rate limiter ignores burst setting'.", "github__create_issue", repo=eq("acme/api"), title=has("rate limiter ignores burst"))
req("gh2", "Comment 'Reproduced on staging' on pull request #482 in acme/web.", "github__add_issue_comment", repo=eq("acme/web"), number=eq(482), body=has("reproduced on staging"))
req("gh3", "Show me the open pull requests in acme/api that target the release-4.12 branch.", "github__list_pull_requests",
    {"github__search_issues": {"query": has("release-4.12")}}, repo=eq("acme/api"), base=eq("release-4.12"))
req("gh4", "Squash-merge PR 377 in acme/web.", "github__merge_pull_request", repo=eq("acme/web"), number=eq(377), method=eq("squash"))
req("gh5", "Where in the acme/api code do we call verify_webhook_signature?", "github__search_code", query=has("verify_webhook_signature"))
req("gh6", "Show me the latest runs of the ci.yml workflow for acme/api.", "github__list_workflow_runs", repo=eq("acme/api"), workflow=has("ci.yml"))
req("gh7", "Read the Dockerfile at the root of acme/web.", "github__get_file_contents", repo=eq("acme/web"), path=has("dockerfile"))
req("gh8", "Create a branch called fix/csv-export in acme/web.", "github__create_branch", repo=eq("acme/web"), branch=eq("fix/csv-export"))
req("gl1", "List the open merge requests for the iOS app (acme-mobile/ios).", "gitlab__list_merge_requests", project=eq("acme-mobile/ios"))
req("gl2", "Retry pipeline 88213 in acme-mobile/android.", "gitlab__retry_pipeline", project=eq("acme-mobile/android"), pipeline_id=eq(88213))
req("gl3", "File an issue in the Android app's repo acme-mobile/android: 'Crash when rotating on checkout'.", "gitlab__create_issue", project=eq("acme-mobile/android"), title=has("crash when rotating"))
req("gl4", "Search the iOS app's GitLab issues for 'push notifications'.", "gitlab__search_issues", query=has("push notification"))
req("li1", "What's the status of ENG-2231?", "linear__get_issue", id=eq("ENG-2231"))
req("li2", "Move ENG-1874 to In Review.", "linear__update_issue", id=eq("ENG-1874"), state=has("review"))
req("li3", "Create a Linear issue for the WEB team: 'Pricing page flickers on Safari'.", "linear__create_issue", team=eq("WEB"), title=has("pricing page flickers"))
req("li4", "Add a comment to ENG-2001 saying 'Blocked on the API change'.", "linear__add_comment", id=eq("ENG-2001"), body=has("blocked on the api change"))
req("li5", "Assign ENG-1999 to priya@acme.io.", "linear__assign_issue",
    {"linear__update_issue": {"id": eq("ENG-1999"), "assignee": eq("priya@acme.io")}}, id=eq("ENG-1999"), assignee=eq("priya@acme.io"))
req("li6", "Which cycles does the MOB team have?", "linear__list_cycles", team=eq("MOB"))
req("ji1", "Show me SUP-4410.", "jira__get_issue", key=eq("SUP-4410"))
req("ji2", "Find support tickets that mention refunds.", "jira__search_issues", jql=has("refund"))
req("ji3", "Close SUP-3982 by moving it to Done.", "jira__transition_issue", key=eq("SUP-3982"), status=eq("done"))
req("ji4", "Reply on SUP-4102: 'We have shipped a fix, please retry.'", "jira__add_comment", key=eq("SUP-4102"), body=has("shipped a fix"))
req("ji5", "Log a new support bug: 'Invoice PDF shows wrong VAT'.", "jira__create_issue", project=eq("SUP"), summary=has("invoice pdf shows wrong vat"))
req("ji6", "Link SUP-4200 to SUP-4185 as 'relates to'.", "jira__link_issues", from_key=eq("SUP-4200"), to_key=eq("SUP-4185"))
req("sl1", "Post 'Deploy of 4.12 is done' in #releases.", "slack__post_message", channel=eq("releases"), text=has("deploy of 4.12 is done"))
req("sl2", "DM maria@acme.io on Slack: 'Can you review my PR?'", "slack__send_direct_message", user=has("maria"), text=has("review my pr"))
req("sl3", "Search Slack for messages about 'db migration' in #backend.", "slack__search_messages", query=has("db migration"))
req("sl4", "Set my Slack status to 'In a workshop' for 2 hours.", "slack__set_status", text=has("in a workshop"), expires_in_minutes=eq(120))
req("sl5", "Schedule 'Standup moved to 10:30' in #team-web for 2026-10-26T09:00:00Z.", "slack__schedule_message", channel=eq("team-web"), text=has("standup moved"), post_at=has("2026-10-26"))
req("sl6", "Create a private Slack channel called incident-4471.", "slack__create_channel", name=eq("incident-4471"), private=eq(True))
req("sl7", "React with :eyes: to message 1729850000.000200 in #support.", "slack__add_reaction", channel=eq("support"), timestamp=eq("1729850000.000200"), emoji=eq("eyes"))
req("sl8", "Read the last 20 messages in #oncall.", "slack__get_channel_history", channel=eq("oncall"), limit=eq(20))
req("tm1", "Post 'Q4 deck is ready' in the General channel of the Sales team.", "teams__send_channel_message", team=has("sales"), channel=has("general"), text=has("q4 deck is ready"))
req("tm2", "Message our partner dana@partner.com: 'Contract signed, thanks!'", "teams__send_chat_message", user=eq("dana@partner.com"), text=has("contract signed"))
req("tm3", "Set up a meeting 'Partner sync' with dana@partner.com on 2026-10-29 from 16:00 to 16:30 UTC.", "teams__create_meeting", subject=has("partner sync"), attendees=st("dana@partner.com"), start=has("2026-10-29"))
req("gm1", "Email finance@acme.io with subject 'October invoices' and body 'All October invoices are in Drive.'", "gmail__send_email", to=st("finance@acme.io"), subject=has("october invoices"))
req("gm2", "Find emails from legal@acme.io that have attachments.", "gmail__search_emails", query=has("legal@acme.io"))
req("gm3", "Draft, but do not send, an email to ceo@acme.io with subject 'Board prep' and body 'Draft agenda attached.'", "gmail__create_draft", to=st("ceo@acme.io"), subject=has("board prep"))
req("gm4", "Forward email 18c2f9a7 to audit@acme.io.", "gmail__forward_email", message_id=eq("18c2f9a7"), to=st("audit@acme.io"))
req("gm5", "Archive email 18c2fa01.", "gmail__archive_email", message_id=eq("18c2fa01"))
req("gm6", "Make a Gmail filter that labels all mail from noreply@github.com as GitHub.", "gmail__create_filter", criteria=has("noreply@github.com"), action=has("label"))
req("gm7", "Reply to all on email 18c2fb10: 'Thanks, confirmed.'", "gmail__reply_to_email", message_id=eq("18c2fb10"), reply_all=eq(True))
req("gc1", "Put 'Design review' on my calendar from 2026-10-28T14:00 to 15:00 with lee@acme.io.", "gcal__create_event", title=has("design review"), start=has("2026-10-28t14:00"), attendees=st("lee@acme.io"))
req("gc2", "What's on my calendar on 2026-10-27?", "gcal__list_events", start=has("2026-10-27"))
req("gc3", "Decline the invitation evt_7Hq2.", "gcal__respond_to_event", event_id=eq("evt_7Hq2"), response=eq("declined"))
req("gc4", "Find a 30-minute slot between 2026-11-02 and 2026-11-06 when ana@acme.io and raj@acme.io are both free.", "gcal__find_free_time", attendees=st("ana@acme.io", "raj@acme.io"), duration_minutes=eq(30))
req("gc5", "Remind me to renew the SSL certificate on 2026-11-15 at 09:00.", "gcal__create_reminder", title=has("ssl"), due=has("2026-11-15"))
req("gc6", "Move event evt_9Kp1 to start at 2026-10-30T11:00 and end at 12:00.", "gcal__update_event", event_id=eq("evt_9Kp1"), start=has("2026-10-30t11:00"))
req("gd1", "Find the 'Q3 board deck' in our shared files.", "gdrive__search_files", query=has("q3 board deck"))
req("gd2", "Share file 1AbC9xYz with ops@acme.io as a commenter.", "gdrive__share_file", file_id=eq("1AbC9xYz"), emails=st("ops@acme.io"), role=eq("commenter"))
req("gd3", "Export file 1Qw3rTy as a PDF.", "gdrive__export_file", file_id=eq("1Qw3rTy"), format=eq("pdf"))
req("gd4", "Create a Google Doc called 'Incident 4471 timeline'.", "gdrive__create_document", title=has("incident 4471 timeline"))
req("no1", "Find the product spec for offline mode.", "notion__search", query=has("offline mode"))
req("no2", "Show entries in the Roadmap database whose Status is In review.", "notion__query_database", database=has("roadmap"), filter=has("in review"))
req("cf1", "Find the runbook for rotating database credentials.", "confluence__search", query=has("credential"))
req("cf2", "Create a postmortem page 'Incident 4471: checkout outage' in the OPS space with the body 'TBD'.", "confluence__create_page", space=eq("OPS"), title=has("incident 4471"))
req("se1", "Show unresolved errors in the web-frontend project.", "sentry__search_issues", query=has("unresolved"), project=eq("web-frontend"))
req("se2", "Resolve error WEB-3F2 in the next release.", "sentry__resolve_issue", issue_id=eq("WEB-3F2"), in_next_release=eq(True))
req("se3", "Show me the latest stack traces for error API-1K9.", "sentry__list_issue_events", issue_id=eq("API-1K9"))
req("se4", "Assign error API-1K9 to team:payments.", "sentry__assign_issue", issue_id=eq("API-1K9"), assignee=has("payments"))
req("dd1", "Show CPU usage for service:api over the last hour.", "datadog__query_metrics", query=has("cpu"), from_time=has("1h"))
req("dd2", "Find error logs from the payments service in the last 15 minutes.", "datadog__search_logs", query=has("payments"), from_time=has("15m"))
req("dd3", "Mute monitor 55012 for 60 minutes.", "datadog__mute_monitor", monitor_id=eq(55012), duration_minutes=eq(60))
req("dd4", "Which monitors are alerting right now?", "datadog__list_monitors", status=eq("alert"))
req("pd1", "Who is on call for the Platform schedule right now?", "pagerduty__get_oncall", schedule=has("platform"))
req("pd2", "Acknowledge incident Q3XK9.", "pagerduty__acknowledge_incident", incident_id=eq("Q3XK9"))
req("pd3", "Page the payments on-call: 'Checkout error rate above 5%', high urgency.", "pagerduty__create_incident", service=has("payments"), urgency=eq("high"))
req("pg1", "Run SELECT count(*) FROM signups WHERE created_at::date = current_date - 1 on the analytics database.", "postgres__query", sql=has("from signups"))
req("pg2", "Show the columns of the orders table.", "postgres__describe_table", table=has("orders"))
req("st1", "Find the billing customer with email jo@example.com.", "stripe__search_customers", query=has("jo@example.com"))
req("st2", "Refund $25 of charge ch_3PqR.", "stripe__create_refund", charge_id=eq("ch_3PqR"), amount_cents=eq(2500))
req("st3", "Cancel subscription sub_9Kd2 at the end of the billing period.", "stripe__cancel_subscription", subscription_id=eq("sub_9Kd2"), at_period_end=eq(True))
req("st4", "List open invoices for cus_77Ab.", "stripe__list_invoices", customer_id=eq("cus_77Ab"), status=eq("open"))
req("hs1", "Add a CRM contact: Lena Ortiz, lena@northwind.com, company Northwind.", "hubspot__create_contact", email=eq("lena@northwind.com"), company=has("northwind"))
req("hs2", "Log a call on contact 9918: 'Discussed renewal pricing'.", "hubspot__log_activity", contact_id=eq("9918"), type=eq("call"))
req("hs3", "Move deal 7720 to Closed Won.", "hubspot__update_deal", deal_id=eq("7720"), stage=has("closed"))
req("fs1", "Find every YAML file in the workspace.", "fs__search_files", pattern=("any", ["yaml", "yml", "y*ml"]))
req("fs2", "Search the workspace files for the text 'TODO(security)'.", "fs__grep", pattern=has("todo"))
req("fs3", "Rename notes.md to notes-2026.md in the workspace.", "fs__move_file", source=has("notes.md"), destination=has("notes-2026.md"))

REQUESTS = R
assert len({r["id"] for r in R}) == len(R)


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower().lstrip("#@")
    return v


def _ok(matcher, value):
    kind, want = matcher
    if value is None:
        return False
    if kind == "eq":
        if isinstance(want, bool):
            return value is want or str(value).lower() == str(want).lower()
        if isinstance(want, int):
            try:
                return int(value) == want
            except (TypeError, ValueError):
                return False
        return _norm(str(value)) == _norm(want)
    if kind == "has":
        text = " ".join(map(str, value)) if isinstance(value, list) else str(value)
        return want.lower() in text.lower()
    if kind == "any":
        return any(w in str(value).lower() for w in want)
    if kind == "set":
        vals = value if isinstance(value, list) else [value]
        return {_norm(str(x)) for x in vals} == {_norm(w) for w in want}
    raise ValueError(kind)


def check(r, name, args):
    """-> (right_tool, right_args)"""
    if name == r["tool"]:
        spec = r["args"]
    elif name in r["alt"]:
        spec = r["alt"][name]
    else:
        return False, False
    return True, all(_ok(m, (args or {}).get(k)) for k, m in spec.items())


if __name__ == "__main__":
    import catalog
    for r in R:
        assert r["tool"] in catalog.BY_NAME, r["tool"]
        props = catalog.BY_NAME[r["tool"]]["input_schema"]["properties"]
        assert set(r["args"]) <= set(props), (r["id"], set(r["args"]) - set(props))
        for a, spec in r["alt"].items():
            assert a in catalog.BY_NAME and set(spec) <= set(catalog.BY_NAME[a]["input_schema"]["properties"])
    from collections import Counter
    print(len(R), "requests;", len({r["tool"] for r in R}), "distinct target tools;",
          Counter(r["tool"].split("__")[0] for r in R))
