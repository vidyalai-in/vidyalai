"""Appends six longer tasks (t25-t30) to tasks.json, with answers computed from the sandbox CSVs.
Run after build_sandbox.py."""
import csv, json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE / "sandbox"
inc = list(csv.DictReader(open(ROOT / "data/incidents.csv")))
roster = list(csv.DictReader(open(ROOT / "data/oncall_roster.csv")))
owner = {"billing-api": "Payments", "refunds-worker": "Payments", "auth-gateway": "Identity",
         "search-indexer": "Platform", "notification-hub": "Growth", "telemetry-ingest": "Data"}
headcount = {"Payments": 7, "Identity": 5, "Platform": 9, "Growth": 6, "Data": 4}
slo = {"billing-api": 99.95, "refunds-worker": None, "auth-gateway": 99.99, "search-indexer": 99.5,
       "notification-hub": 99.9, "telemetry-ingest": 99.9}

big = sum(int(r["minutes_down"]) for r in inc if headcount[owner[r["service"]]] > 5)
team12 = Counter(owner[r["service"]] for r in inc if r["severity"] in ("SEV1", "SEV2"))
(top_team, top_n), (_, second_n) = team12.most_common(2)
assert top_n != second_n
strict = [s for s, v in slo.items() if v is not None and v >= 99.9]
q3 = sum(int(r["minutes_down"]) for r in inc if r["service"] in strict and "2026-07-01" <= r["date"] <= "2026-09-30")
last_sev1 = max((r for r in inc if r["severity"] == "SEV1"), key=lambda r: r["date"])
d = date.fromisoformat(last_sev1["date"])
wk = date.fromordinal(d.toordinal() - d.weekday()).isoformat()
team = owner[last_sev1["service"]]
prim = [r["primary"] for r in roster if r["team"] == team and r["week_start"] == wk]
assert prim, f"latest SEV1 {last_sev1} not in roster"
per_service = Counter(r["service"] for r in inc)
least = min(per_service.values())
assert list(per_service.values()).count(least) == 1
least_svc = min(per_service, key=per_service.get)

T = [t for t in json.loads((HERE / "tasks.json").read_text()) if t["kind"] != "long"]
def task(q, *groups):
    T.append({"id": f"t{len(T)+1:02d}", "kind": "long", "question": q, "must": [list(g) for g in groups]})

task("Across the whole incident log, what is the total number of minutes down for services owned by teams with more than 5 engineers?", [str(big), f"{big:,}"])
task("Which team owns the services with the most SEV1 and SEV2 incidents combined, and how many is that?", [top_team], [str(top_n)])
task("Take every service whose availability SLO is 99.9% or stricter. What is their combined number of minutes down in Q3 2026 (July to September)?", [str(q3), f"{q3:,}"])
task("Find the runbook for the service owned by the team with the smallest headcount. What lag threshold pages the on-call engineer?", ["120 seconds", "120"])
task("Who was the primary on-call engineer for the owning team in the week of the most recent SEV1 incident?", [prim[0]])
task("Which service has the fewest incidents in the log, and who leads the team that owns it?", [least_svc], [{"Payments": "Priya Raman", "Identity": "Tomás Ferreira", "Platform": "Grace Okafor", "Growth": "Daniel Kim", "Data": "Mei Lin Chen"}[owner[least_svc]]])
(HERE / "tasks.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
for t in T[-6:]: print(t["id"], t["must"], "|", t["question"])
print("latest SEV1:", last_sev1, "week", wk, "| per service:", dict(per_service), "| team SEV1+2:", dict(team12))
