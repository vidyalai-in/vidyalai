"""Exp 5: tool-selection accuracy with terse vs detailed tool descriptions (Haiku 5.5).

usage: python exp5_descriptions.py <repeats>
"""
import sys, time, random
from concurrent.futures import ThreadPoolExecutor
from common import ledger, HAIKU, dump

cl = ledger.client()
Q = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# name: (terse description, detailed description)
DESCS = {
    "helpscout_search": ("Search Help Scout.",
        "Search the public help-center articles we publish for customers (how-to guides, FAQs, setup steps). "
        "Use it to answer how a feature is supposed to work from the customer's point of view. "
        "It does not contain past support conversations or internal engineering material."),
    "zendesk_search": ("Search Zendesk.",
        "Search past customer support tickets and their replies. Use it to find whether customers have reported "
        "a problem before, how often, and what support told them. It does not contain official documentation."),
    "confluence_search": ("Search Confluence.",
        "Search the internal engineering wiki: runbooks, on-call procedures, architecture and design documents, "
        "postmortems. Use it for how our systems are built or operated. Customers cannot see these pages."),
    "datadog_metrics": ("Query Datadog.",
        "Query time-series metrics such as request latency, error rates, queue depth, daily active users and "
        "signups, aggregated per minute or per day. Use it for numbers over time, not for individual log lines."),
    "loki_logs": ("Query Loki.",
        "Search raw application log lines by service, time range and text, for example a specific error message "
        "or a request ID. Use it to inspect individual events, not aggregate trends."),
    "stripe_customer": ("Look up a Stripe customer.",
        "Look up a customer's billing record: subscription plan, payment method, invoices, refunds and failed "
        "charges. Use it for anything about what a customer pays or was charged."),
    "statuspage_incidents": ("List Statuspage incidents.",
        "List public incidents and outages we announced to customers, with start and end times and affected "
        "components. Use it to check whether there was an outage at a given time."),
    "linear_search": ("Search Linear.",
        "Search engineering issues: open bugs, feature requests and planned roadmap work, with status and owner. "
        "Use it to find whether a bug is known to engineering or a feature is planned. It does not list shipped releases."),
    "github_releases": ("List GitHub releases.",
        "List release notes for versions we have shipped, with dates and the changes in each. Use it to find "
        "when a feature or fix went live."),
    "pagerduty_oncall": ("Query PagerDuty.",
        "Find who is on call right now or at a given time for a team or service, and the escalation policy. "
        "Use it to find the person to page; it has no general employee directory."),
}


def tools(detailed):
    return [{"name": n, "description": d[1] if detailed else d[0], "input_schema": Q} for n, d in DESCS.items()]


TASKS = [
    ("helpscout_search", "A customer asks how to set up SAML single sign-on. What steps do we tell them?"),
    ("helpscout_search", "Where do customers go to read how to export their data to CSV?"),
    ("helpscout_search", "What does our official guide say about inviting teammates to a workspace?"),
    ("zendesk_search", "Have other customers complained that the CSV export cuts off rows?"),
    ("zendesk_search", "What did support tell Acme Corp last month when they asked about data residency?"),
    ("zendesk_search", "How many people wrote in this week about not receiving password reset emails?"),
    ("confluence_search", "What's the procedure for restarting the ingest workers safely?"),
    ("confluence_search", "Is there a postmortem for the March database failover?"),
    ("confluence_search", "How is the billing service designed to handle webhook retries?"),
    ("datadog_metrics", "What was the p95 API latency over the last 24 hours?"),
    ("datadog_metrics", "How many daily active users did we have each day last week?"),
    ("datadog_metrics", "Is the error rate on checkout higher today than yesterday?"),
    ("loki_logs", "Find the error message for request ID 7f3a-22b1 that failed at 14:02."),
    ("loki_logs", "Show me the stack trace the payments service printed when it crashed this morning."),
    ("loki_logs", "Which exact lines did the auth service log for user 4411 around noon?"),
    ("stripe_customer", "Why was customer 8812 charged twice in October?"),
    ("stripe_customer", "Which plan is Globex on, and is their card about to expire?"),
    ("stripe_customer", "Did we already refund invoice 2231 for Initech?"),
    ("statuspage_incidents", "Was there an outage yesterday afternoon that customers were told about?"),
    ("statuspage_incidents", "When did last Tuesday's degraded performance incident end?"),
    ("statuspage_incidents", "Which components were affected in the most recent announced incident?"),
    ("linear_search", "Does engineering already know about the bug where dark mode resets on reload?"),
    ("linear_search", "Is offline mode for the mobile app on the roadmap?"),
    ("linear_search", "Who owns the open issue about slow dashboard loads?"),
    ("github_releases", "When did we ship SSO for the Team plan?"),
    ("github_releases", "What changed in version 4.12?"),
    ("github_releases", "Has the fix for the timezone bug in reports gone live yet?"),
    ("pagerduty_oncall", "Who should I page about the payments service right now?"),
    ("pagerduty_oncall", "Who was on call for the data platform last Saturday night?"),
    ("pagerduty_oncall", "What's the escalation path if the search team's primary doesn't respond?"),
]
SYSTEM = "You are an internal assistant for a software company. Use one tool to look up the answer."


def run(job):
    detailed, idx, rep = job
    expected, prompt = TASKS[idx]
    t0 = time.time()
    r = ledger.create(cl, "a02", note=f"exp5 detailed={detailed}", model=HAIKU, max_tokens=2048, system=SYSTEM,
                      tools=tools(detailed), messages=[{"role": "user", "content": prompt}])
    names = [b.name for b in r.content if b.type == "tool_use"]
    rec = {"detailed": detailed, "idx": idx, "rep": rep, "expected": expected, "called": names,
           "first_ok": bool(names) and names[0] == expected, "secs": round(time.time() - t0, 3),
           "usage": r.usage.model_dump(), "stop": r.stop_reason}
    dump("exp5_descriptions.jsonl", rec)
    return rec


if __name__ == "__main__":
    reps = int(sys.argv[1])
    jobs = [(d, i, k) for k in range(reps) for i in range(len(TASKS)) for d in (False, True)]
    random.seed(5)
    random.shuffle(jobs)
    with ThreadPoolExecutor(6) as ex:
        for rec in ex.map(run, jobs):
            if not rec["first_ok"]:
                print("detailed" if rec["detailed"] else "terse   ", rec["idx"], rec["expected"], "->", rec["called"])
    print(ledger.report())
