"""Builds the Part 1 sandbox: a small internal-docs folder for a fictional company, plus tasks.json.

Run: python build_sandbox.py   (writes ./sandbox/ and ./tasks.json next to this file)
Aggregation answers are computed from the generated CSVs, so the answer key is correct by construction.
"""
import csv, json, random, shutil
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE / "sandbox"
shutil.rmtree(ROOT, ignore_errors=True)

def w(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n")

w("README.md", """
# Larkspur Robotics internal docs

Engineering handbook for Larkspur Robotics (fictional). Folders:

- docs/services/: one page per production service
- docs/teams/: one page per engineering team
- docs/runbooks/: incident runbooks
- docs/policies/: company policies (travel, expenses, on-call, time off)
- docs/changelog.md: platform release notes
- data/: CSV exports (incident log, on-call roster)
""")

SERVICES = {
"billing-api": ("Payments", """
# billing-api

Creates invoices and charges customer payment methods.

- Owner team: Payments
- Language: Go
- Listens on port 8443 (HTTPS)
- Availability SLO: 99.95% per calendar month
- Depends on: auth-gateway, ledger-db
- Paging rotation: payments-primary

Every charge carries an idempotency key; retries with the same key never charge twice.
"""),
"refunds-worker": ("Payments", """
# refunds-worker

Background worker that processes refund requests from the `refunds.v2` queue.

- Owner team: Payments
- Language: Python
- Metrics port 9102
- SLO: 95% of refunds processed within 15 minutes of the request
- Depends on: billing-api
- Runbook: docs/runbooks/refund-backlog.md
"""),
"auth-gateway": ("Identity", """
# auth-gateway

Issues and validates access tokens for every internal and public API.

- Owner team: Identity
- Language: Rust
- Listens on port 8080
- Availability SLO: 99.99% per calendar month
- Access tokens live 20 minutes; refresh tokens live 14 days
- Runbook: docs/runbooks/token-expiry-storm.md
"""),
"search-indexer": ("Platform", """
# search-indexer

Builds the product-search indices from the catalog and the object store.

- Owner team: Platform
- Language: Java
- Listens on port 9200
- Availability SLO: 99.5% per calendar month
- Full reindex runs nightly at 02:30 UTC
- Depends on: object-store
- Runbook: docs/runbooks/indexer-disk-full.md
"""),
"notification-hub": ("Growth", """
# notification-hub

Sends email and mobile push notifications on behalf of other services.

- Owner team: Growth
- Language: TypeScript
- Listens on port 7001
- Availability SLO: 99.9% per calendar month
- Rate limit: 1,200 messages per minute per tenant
- Depends on: auth-gateway, telemetry-ingest
"""),
"telemetry-ingest": ("Data", """
# telemetry-ingest

Receives OpenTelemetry traces, metrics and logs from every service.

- Owner team: Data
- Language: Go
- Listens on port 4317 (OTLP over gRPC)
- Availability SLO: 99.9% per calendar month
- Raw events are kept for 30 days, then aggregated
- Depends on: object-store
- Runbook: docs/runbooks/telemetry-lag.md
"""),
}
for name, (_, text) in SERVICES.items():
    w(f"docs/services/{name}.md", text)

TEAMS = {
"Payments": ("payments", "Priya Raman", "#pay-oncall", ["Priya Raman", "Arjun Mehta", "Sofia Rossi", "Kwame Asante", "Lena Vogel", "Hiro Tanaka", "Nadia Haddad"]),
"Identity": ("identity", "Tomás Ferreira", "#idn-alerts", ["Tomás Ferreira", "Aisha Bello", "Jonas Berg", "Mira Patel", "Oscar Lindqvist"]),
"Platform": ("platform", "Grace Okafor", "#plat-ops", ["Grace Okafor", "Ravi Iyer", "Elena Petrova", "Sam Whitaker", "Yuki Mori", "Diego Alvarez", "Fatima Zahra", "Ben Cohen", "Ivy Nguyen"]),
"Growth": ("growth", "Daniel Kim", "#growth-eng", ["Daniel Kim", "Chloe Martin", "Omar Farouk", "Leah Goldberg", "Pedro Santos", "Anya Ivanova"]),
"Data": ("data", "Mei Lin Chen", "#data-infra", ["Mei Lin Chen", "Luca Bianchi", "Zara Ahmed", "Felix Wagner"]),
}
for team, (slug, lead, chan, members) in TEAMS.items():
    owned = [s for s, (t, _) in SERVICES.items() if t == team]
    w(f"docs/teams/{slug}.md", f"""
# {team} team

- Team lead: {lead}
- Slack channel: {chan}
- Headcount: {len(members)} engineers
- Members: {", ".join(members)}
- Services owned: {", ".join(owned)}

Escalations go to the team lead after the on-call engineer has worked an incident for 30 minutes.
""")

w("docs/runbooks/refund-backlog.md", """
# Runbook: refund backlog

Applies to: refunds-worker

1. The on-call engineer is paged when the `refunds.v2` queue depth stays above 5,000 messages for 10 minutes.
2. Scale refunds-worker to 12 replicas.
3. Check billing-api latency; refunds stall when billing-api p99 latency exceeds 800 ms.
4. If the backlog is still growing after 30 minutes, escalate to the owning team's lead.
""")
w("docs/runbooks/token-expiry-storm.md", """
# Runbook: token expiry storm

Applies to: auth-gateway

1. The on-call engineer is paged when HTTP 401 responses exceed 3% of requests for 5 minutes.
2. Confirm clocks are in sync on the gateway hosts (skew above 2 seconds causes early expiry).
3. As a mitigation, extend the access-token lifetime to 60 minutes with the flag `auth.token_ttl_override`.
4. Remove the override within 24 hours.
""")
w("docs/runbooks/indexer-disk-full.md", """
# Runbook: indexer disk full

Applies to: search-indexer

1. An alert fires when any indexer node passes 85% disk usage; the on-call engineer is paged at 92%.
2. Delete index snapshots older than 7 days.
3. Never delete indices whose names start with `live-`.
4. If usage stays above 85%, add a node to the cluster.
""")
w("docs/runbooks/telemetry-lag.md", """
# Runbook: telemetry lag

Applies to: telemetry-ingest

1. The on-call engineer is paged when ingest lag exceeds 120 seconds for 10 minutes.
2. Drop debug-level events first by setting `ingest.min_level=info`.
3. If lag persists, add ingest replicas (maximum 24).
""")

w("docs/policies/travel.md", """
# Travel policy

Hotel cap per night:

| City | Cap per night |
|---|---|
| London | GBP 210 |
| New York | USD 320 |
| Singapore | SGD 280 |
| Any other city | USD 180 |

Flights: economy class for flights under 6 hours; premium economy is allowed for flights of 6 hours or more.
Book through the travel portal at least 14 days before departure.
""")
w("docs/policies/expenses.md", """
# Expense policy

- Meals: up to USD 75 per day on domestic trips, USD 95 per day on international trips.
- Receipts are required for any single expense above USD 25.
- Submit expenses within 30 days of the expense date; later claims need director approval.
""")
w("docs/policies/oncall.md", """
# On-call policy

- Primary on-call receives a stipend of USD 400 per week; secondary receives USD 150 per week.
- Weeks start on Monday. The roster is exported weekly to data/oncall_roster.csv.
- Anyone paged for a SEV1 between 00:00 and 06:00 local time gets one day off in lieu.
- Acknowledge a page within 5 minutes; SEV1 incidents need a written review within 3 business days.
""")
w("docs/policies/pto.md", """
# Time off

- 25 days of paid time off per calendar year.
- At most 5 unused days carry over into the next year.
- New joiners accrue 2.08 days per month in their first year.
""")
w("docs/changelog.md", """
# Platform changelog

## v4.5.0 (2026-09-08)
- notification-hub: per-tenant rate limit raised from 900 to 1,200 messages per minute.

## v4.4.0 (2026-07-19)
- search-indexer: nightly full reindex moved from 01:00 UTC to 02:30 UTC.

## v4.3.0 (2026-05-02)
- auth-gateway: access-token lifetime changed from 15 minutes to 20 minutes.

## v4.2.0 (2026-03-11)
- refunds-worker: switched to the `refunds.v2` queue.
""")

# ---- data/incidents.csv (generated, seeded) ----
rng = random.Random(8)
svc_names = list(SERVICES)
weights = [6, 4, 5, 5, 6, 4]
rows = []
d = date(2026, 1, 3)
for i in range(40):
    d += timedelta(days=rng.randint(3, 9))
    svc = rng.choices(svc_names, weights)[0]
    sev = rng.choices(["SEV1", "SEV2", "SEV3"], [2, 5, 6])[0]
    mins = {"SEV1": rng.randint(25, 140), "SEV2": rng.randint(8, 60), "SEV3": rng.randint(0, 15)}[sev]
    summary = rng.choice(["elevated error rate", "latency spike", "dependency timeout", "bad deploy rolled back",
                          "certificate expiry", "queue backlog", "disk pressure", "config typo"])
    rows.append({"id": f"INC-{1040 + i}", "date": d.isoformat(), "service": svc, "severity": sev,
                 "minutes_down": mins, "summary": summary})
(ROOT / "data").mkdir(parents=True, exist_ok=True)
with open(ROOT / "data/incidents.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)

# ---- data/oncall_roster.csv ----
roster = []
wk = date(2026, 7, 6)
for k in range(13):
    for team, (_, _, _, members) in TEAMS.items():
        prim = members[(k * 2) % len(members)]
        sec = members[(k * 2 + 1) % len(members)]
        roster.append({"week_start": (wk + timedelta(weeks=k)).isoformat(), "team": team, "primary": prim, "secondary": sec})
with open(ROOT / "data/oncall_roster.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(roster[0]))
    wr.writeheader(); wr.writerows(roster)

# ---- answers computed from the data ----
owner = {s: t for s, (t, _) in SERVICES.items()}
sev1 = sum(r["severity"] == "SEV1" for r in rows)
q2 = defaultdict(int)
for r in rows:
    if "2026-04-01" <= r["date"] <= "2026-06-30":
        q2[r["service"]] += r["minutes_down"]
q2_top = max(q2, key=q2.get)
assert sorted(q2.values())[-1] != sorted(q2.values())[-2], "tie in Q2"
billing_min = sum(r["minutes_down"] for r in rows if r["service"] == "billing-api")
pay_n = sum(owner[r["service"]] == "Payments" for r in rows)
months = Counter(r["date"][:7] for r in rows)
top_month, top_n = months.most_common(1)[0]
assert list(months.values()).count(top_n) == 1, "tie in months"
grace = sum(r["primary"] == "Grace Okafor" for r in roster)
idn_max = max(r["minutes_down"] for r in rows if owner[r["service"]] == "Identity")
auth_after = sum(r["service"] == "auth-gateway" and r["date"] > "2026-05-02" for r in rows)
assert auth_after > 0
data_prim = next(r["primary"] for r in roster if r["team"] == "Data" and r["week_start"] == "2026-08-24")
month_name = date(2026, int(top_month[5:]), 1).strftime("%B")

# Each task: id, kind, question, must = list of alternative-groups; every group needs one match.
T = []
def task(kind, q, *groups):
    T.append({"id": f"t{len(T)+1:02d}", "kind": kind, "question": q, "must": [list(g) for g in groups]})

task("lookup", "What port does search-indexer listen on?", ["9200"])
task("lookup", "What is the per-night hotel cap for a business trip to Singapore?", ["280"])
task("lookup", "Within how many days of the expense date must expenses be submitted?", ["30"])
task("lookup", "At what disk usage percentage does the search-indexer disk alert fire?", ["85"])
task("lookup", "In which release did the access-token lifetime change to 20 minutes?", ["v4.3.0", "4.3.0"])
task("lookup", "What is the weekly stipend for primary on-call?", ["400"])
task("lookup", "How long does telemetry-ingest keep raw events?", ["30 days", "30"])
task("lookup", "Which Slack channel does the Data team use?", ["#data-infra", "data-infra"])
task("multi-hop", "Who leads the team that owns refunds-worker?", ["Priya Raman"])
task("multi-hop", "Which Slack channel should you post in when auth-gateway's SLO is breached?", ["#idn-alerts", "idn-alerts"])
task("multi-hop", "What queue-depth threshold pages the on-call engineer for the service that depends directly on billing-api?", ["5000", "5,000"])
task("multi-hop", "How many engineers are on the team that owns the service listening on port 7001?", ["6", "six"])
task("multi-hop", "The nightly reindex now runs at 02:30 UTC. In which release did that change, and what time did it run before?", ["v4.4.0", "4.4.0"], ["01:00", "1:00", "1 am", "1am"])
task("multi-hop", "Who leads the team responsible for the service with the strictest availability SLO?", ["Tomás Ferreira", "Tomas Ferreira"])
task("multi-hop", "What is the current per-tenant rate limit of the service that depends on both auth-gateway and telemetry-ingest, and what was it before?", ["1200", "1,200"], ["900"])
task("multi-hop", "Who was primary on-call for the team that owns telemetry-ingest in the week starting 2026-08-24?", [data_prim])
task("aggregate", "How many SEV1 incidents are in the incident log?", [str(sev1)])
task("aggregate", "Which service had the most total minutes down in Q2 2026 (April to June)?", [q2_top])
task("aggregate", "What is the total number of minutes down for billing-api across the whole incident log?", [str(billing_min), f"{billing_min:,}"])
task("aggregate", "How many incidents in the log involve services owned by the Payments team?", [str(pay_n)])
task("aggregate", "In which month of 2026 did the most incidents happen?", [month_name, top_month])
task("aggregate", "How many weeks was Grace Okafor the primary on-call in the roster?", [str(grace)])
task("aggregate", "What was the longest single outage, in minutes, for a service owned by the Identity team?", [str(idn_max)])
task("aggregate", "How many auth-gateway incidents happened after the release that changed the access-token lifetime to 20 minutes?", [str(auth_after)])

(HERE / "tasks.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
print(len(T), "tasks;", sum(1 for _ in ROOT.rglob("*") if _.is_file()), "files")
for t in T: print(t["id"], t["kind"], t["must"], "|", t["question"])
