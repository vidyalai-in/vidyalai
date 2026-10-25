"""Part 5 test bed: Larkspur Robotics' incident tracker, generated deterministically.

About 1,000 incident records of roughly 2,300 tokens each (header, summary, timeline, a 28-line log
excerpt, impact, action items). Fourteen chains of records (each record names its follow-up incident)
and eight reserved (service, month) groups carry the 22 tasks; every answer is computed here.

python corpus.py  -> writes incidents.json and tasks.json next to this file.
"""
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
R = random.Random(20261025)

SERVICES = {
    "fleet-telemetry": "Data Platform", "route-planner": "Navigation", "charge-scheduler": "Energy",
    "map-tiles": "Navigation", "auth-gateway": "Identity", "billing-api": "Payments",
    "notification-hub": "Growth", "firmware-ota": "Devices", "vision-inference": "Perception",
    "dock-controller": "Energy", "payments-ledger": "Payments", "search-indexer": "Data Platform",
    "battery-health": "Devices", "order-intake": "Growth", "lidar-calibration": "Perception",
    "ops-console": "Internal Tools",
}
CAUSES = ["config-change", "dependency-outage", "capacity", "bad-deploy", "certificate-expiry",
          "data-corruption", "network-partition", "hardware-fault"]
SEVS = ["SEV1", "SEV2", "SEV3", "SEV4"]
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09"]
CODES = ["E401", "E404", "E409", "E429", "E500", "E502", "E503", "E504", "E507"]
COMPONENTS = ["ingest", "scheduler", "api", "worker", "cache", "db-pool", "grpc", "queue", "auth", "sync"]
SYMPTOMS = ["Elevated 5xx rate", "Latency spike", "Stuck job queue", "Partial outage",
            "Delayed updates", "Failed rollouts", "Timeouts to upstream", "Error budget burn",
            "Degraded throughput", "Intermittent failures"]
VERBS = ["acknowledged the page", "opened a bridge call", "rolled back the last change",
         "scaled the worker pool", "drained the affected zone", "paused the rollout",
         "rotated the credentials", "failed over to the standby", "posted a status update",
         "confirmed recovery on dashboards"]


def hexid(n=5):
    return "".join(R.choice("0123456789abcdef") for _ in range(n))


def log_lines(service, start, first_code, n=28):
    """A realistic-looking log excerpt. The first ERROR line carries `first_code`."""
    lines, t, first_done = [], start - timedelta(minutes=2), False
    pod = f"{service}-{hexid(4)}{hexid(1)}"
    for i in range(n):
        t += timedelta(seconds=R.randint(1, 40), milliseconds=R.randint(0, 999))
        comp = R.choice(COMPONENTS)
        roll = R.random()
        if not first_done and i >= R.randint(2, 6):
            lvl, msg, first_done = "ERROR", f"request failed code={first_code} upstream={R.choice(list(SERVICES))} req={hexid(8)}", True
        elif first_done and roll < 0.25:
            lvl, msg = "ERROR", f"request failed code={R.choice(CODES)} upstream={R.choice(list(SERVICES))} req={hexid(8)}"
        elif roll < 0.55:
            lvl, msg = "WARN", f"retry {R.randint(1, 5)}/5 for batch b-{R.randint(10000, 99999)} after {R.choice([250, 500, 1000, 1500, 3000])}ms (queue_depth={R.randint(10, 9000)})"
        else:
            lvl, msg = "INFO", f"processed {R.randint(1, 900)} items in {R.randint(3, 4000)}ms shard={R.randint(0, 63)} node={hexid(6)}"
        lines.append(f"{t.strftime('%Y-%m-%dT%H:%M:%S.')}{t.microsecond // 1000:03d}Z {pod} {lvl:5s} [{comp}] {msg}")
    return lines


def make_incident(iid, service, month, follow="none", sev=None):
    day = R.randint(1, 28)
    opened = datetime.strptime(f"{month}-{day:02d}", "%Y-%m-%d") + timedelta(minutes=R.randint(0, 1439))
    duration = R.randint(17, 400)
    resolved = opened + timedelta(minutes=duration)
    sev = sev or R.choices(SEVS, weights=[2, 4, 5, 3])[0]
    cause = R.choice(CAUSES)
    first_code = R.choice(CODES)
    rec = {
        "id": iid, "service": service, "team": SERVICES[service], "severity": sev,
        "opened": opened.strftime("%Y-%m-%d %H:%M"), "resolved": resolved.strftime("%Y-%m-%d %H:%M"),
        "duration_min": duration, "follow": follow, "title": f"{R.choice(SYMPTOMS)} on {service}",
        "impact_min": R.randint(3, 180), "robots": R.randint(12, 4800), "cause": cause,
        "first_code": first_code,
    }
    timeline = []
    t = opened
    for v in R.sample(VERBS, 6):
        t += timedelta(minutes=R.randint(2, max(3, duration // 6)))
        timeline.append(f"{t.strftime('%H:%M')}  On-call {v}.")
    rec["timeline"] = timeline
    rec["log"] = log_lines(service, opened, first_code)
    rec["summary"] = (f"Between {rec['opened']} and {rec['resolved']} UTC, {service} showed "
                      f"{rec['title'].split(' on ')[0].lower()}. Alerts fired from the {R.choice(COMPONENTS)} "
                      f"component and the {SERVICES[service]} on-call engineer responded. "
                      f"Downstream callers saw retries and some requests were dropped before mitigation.")
    rec["actions"] = [f"Add alert on {R.choice(COMPONENTS)} saturation (owner: {SERVICES[service]})",
                      f"Write a runbook entry for {rec['title'].split(' on ')[0].lower()}",
                      f"Review {R.choice(['timeouts', 'retry budgets', 'capacity plans', 'rollout gates'])} for {service}"]
    return rec


def render(rec):
    """The full text a naive get_incident returns."""
    return "\n".join([
        f"INCIDENT {rec['id']}",
        f"Title: {rec['title']}",
        f"Service: {rec['service']} (owner: {rec['team']})",
        f"Severity: {rec['severity']}",
        f"Opened: {rec['opened']} UTC    Resolved: {rec['resolved']} UTC",
        "Status: resolved",
        f"Follow-up incident: {rec['follow']}",
        "",
        "Summary",
        rec["summary"],
        "",
        "Timeline (UTC)",
        *rec["timeline"],
        "",
        f"Log excerpt ({len(rec['log'])} lines)",
        *rec["log"],
        "",
        "Impact",
        f"Customer impact minutes: {rec['impact_min']}",
        f"Robots affected: {rec['robots']:,}",
        f"Root cause category: {rec['cause']}",
        "",
        "Action items",
        *[f"- {a}" for a in rec["actions"]],
    ])


def header(rec):
    return f"{rec['id']} | {rec['opened'][:10]} | {rec['service']} | {rec['severity']} | {rec['title']}"


# ---- tasks -------------------------------------------------------------------------------------

CHAIN_SPECS = [  # (id, length, question kind)
    ("c01", 24, "sum_impact"), ("c02", 22, "count_sev1"), ("c03", 26, "max_robots"),
    ("c04", 28, "mode_cause"), ("c05", 20, "sum_robots"), ("c06", 30, "count_cause"),
    ("c07", 25, "sum_impact_sev12"), ("c08", 27, "distinct_services"), ("c09", 23, "max_duration"),
    ("c10", 32, "sum_impact"), ("c11", 21, "mode_first_code"), ("c12", 29, "max_impact"),
    ("c13", 48, "sum_impact"), ("c14", 48, "max_robots"),  # long: the naive loop passes 100,000 tokens
]
FAN_SPECS = [  # (id, service, month, question kind, size)
    ("f01", "fleet-telemetry", "2026-08", "sum_impact", 24), ("f02", "auth-gateway", "2026-09", "mode_cause", 26),
    ("f03", "route-planner", "2026-07", "max_robots", 22), ("f04", "map-tiles", "2026-08", "count_cause", 28),
    ("f05", "billing-api", "2026-09", "sum_robots_sev12", 25), ("f06", "charge-scheduler", "2026-07", "sum_impact", 21),
    ("f07", "dock-controller", "2026-08", "count_first_code", 23), ("f08", "firmware-ota", "2026-09", "max_impact", 27),
]
MONTH_NAME = {"2026-06": "June 2026", "2026-07": "July 2026", "2026-08": "August 2026", "2026-09": "September 2026"}


def mode(values):
    c = Counter(values).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None  # None: tie, regenerate


def answer(kind, recs, arg=None):
    if kind.endswith("_sev12"):
        recs = [r for r in recs if r["severity"] in ("SEV1", "SEV2")]
    if kind.startswith("sum_impact"):
        return sum(r["impact_min"] for r in recs)
    if kind.startswith("sum_robots"):
        return sum(r["robots"] for r in recs)
    if kind == "count_sev1":
        return sum(r["severity"] == "SEV1" for r in recs)
    if kind == "count_cause":
        return sum(r["cause"] == arg for r in recs)
    if kind == "count_first_code":
        return sum(r["first_code"] == arg for r in recs)
    if kind == "distinct_services":
        return len({r["service"] for r in recs})
    if kind == "mode_cause":
        return mode([r["cause"] for r in recs])
    if kind == "mode_first_code":
        return mode([r["first_code"] for r in recs])
    for key, k in (("max_robots", "robots"), ("max_duration", "duration_min"), ("max_impact", "impact_min")):
        if kind == key:
            top = sorted(recs, key=lambda r: -r[k])
            return top[0]["id"] if top[0][k] != top[1][k] else None
    raise ValueError(kind)


def question(kind, scope, arg=None):
    q = {
        "sum_impact": f"What is the total number of customer impact minutes across {scope}?",
        "sum_impact_sev12": f"What is the total number of customer impact minutes across the SEV1 and SEV2 incidents among {scope}?",
        "sum_robots": f"What is the total number of robots affected across {scope}?",
        "sum_robots_sev12": f"What is the total number of robots affected across the SEV1 and SEV2 incidents among {scope}?",
        "count_sev1": f"How many of {scope} are SEV1?",
        "count_cause": f"How many of {scope} have the root cause category {arg}?",
        "count_first_code": f"In how many of {scope} is the first ERROR line of the log excerpt a code={arg} error?",
        "distinct_services": f"How many distinct services appear among {scope}?",
        "mode_cause": f"Which root cause category is the most common among {scope}?",
        "mode_first_code": f"Each incident's log excerpt has a first ERROR line with an error code (code=E...). Which code is most often the first error among {scope}?",
        "max_robots": f"Which incident among {scope} affected the most robots? Give its incident ID.",
        "max_duration": f"Which incident among {scope} took the longest from opened to resolved? Give its incident ID.",
        "max_impact": f"Which incident among {scope} had the most customer impact minutes? Give its incident ID.",
    }[kind]
    return q


def build():
    ids = R.sample(range(1000, 9999), 1200)
    next_id = iter(f"INC-{i}" for i in ids)
    reserved = {(s, m) for _, s, m, _, _ in FAN_SPECS}
    free = [(s, m) for s in SERVICES for m in MONTHS if (s, m) not in reserved]
    incidents, tasks = {}, []

    for tid, length, kind in CHAIN_SPECS:
        while True:
            chain_ids = [next(next_id) for _ in range(length)]
            recs = []
            for i, iid in enumerate(chain_ids):
                s, m = R.choice(free)
                recs.append(make_incident(iid, s, m, follow=chain_ids[i + 1] if i + 1 < length else "none"))
            arg = None
            if kind == "count_cause":
                arg = Counter(r["cause"] for r in recs).most_common(3)[-1][0]
            ans = answer(kind, recs, arg)
            if ans is not None:
                break
        for r in recs:
            incidents[r["id"]] = r
        scope = (f"the chain of incidents that starts at {chain_ids[0]} (each incident names its follow-up "
                 f"incident; follow the chain until the follow-up is 'none', and include the first and last incident)")
        tasks.append({"id": tid, "kind": "chain", "qkind": kind, "size": length, "answer": ans,
                      "question": question(kind, scope, arg), "members": chain_ids})

    for tid, service, month, kind, size in FAN_SPECS:
        while True:
            recs = [make_incident(next(next_id), service, month) for _ in range(size)]
            arg = Counter(r["first_code"] for r in recs).most_common(2)[-1][0] if kind == "count_first_code" else None
            if kind == "count_cause":
                arg = Counter(r["cause"] for r in recs).most_common(2)[-1][0]
            ans = answer(kind, recs, arg)
            if ans is not None:
                break
        for r in recs:
            incidents[r["id"]] = r
        scope = f"all incidents on the {service} service opened in {MONTH_NAME[month]}"
        tasks.append({"id": tid, "kind": "fanout", "qkind": kind, "size": size, "answer": ans,
                      "question": question(kind, scope, arg), "members": [r["id"] for r in recs]})

    while len(incidents) < 1000:  # background incidents, none in a reserved (service, month)
        s, m = R.choice(free)
        r = make_incident(next(next_id), s, m)
        incidents[r["id"]] = r
    return incidents, tasks


if __name__ == "__main__":
    incidents, tasks = build()
    (HERE / "incidents.json").write_text(json.dumps(incidents))
    (HERE / "tasks.json").write_text(json.dumps(tasks, indent=1))
    lens = [len(render(r)) for r in incidents.values()]
    print(len(incidents), "incidents; chars per record min/mean/max", min(lens), sum(lens) // len(lens), max(lens))
    for t in tasks:
        print(t["id"], t["size"], t["answer"], "|", t["question"][:110])
