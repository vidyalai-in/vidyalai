"""Tools over the Larkspur incident tracker, in three result-sizing designs, plus grading."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from corpus import header, render  # noqa: E402
from harness import tool  # noqa: E402

INCIDENTS = json.loads((HERE / "incidents.json").read_text())
TASKS = json.loads((HERE / "tasks.json").read_text())
ORDER = sorted(INCIDENTS.values(), key=lambda r: (r["opened"], r["id"]))

SYSTEM = """You are an incident analyst at Larkspur Robotics. Answer the question using the incident
tracker tools. Read every record the question needs and never guess a value you have not read.
When you are done, end your reply with one line in the form
ANSWER: <the answer only: a number, an incident ID or a category>"""

ID = {"type": "string", "description": "An incident ID such as INC-4821"}
QUERY = {"type": "string", "description": "Case-insensitive text or Python regular expression matched "
         "against each incident's header line 'ID | opened date | service | severity | title', "
         "e.g. 'fleet-telemetry' or '2026-08-.*map-tiles'"}


def _search(query):
    rx = re.compile(query, re.IGNORECASE)
    return [header(r) for r in ORDER if rx.search(header(r))]


def _get(incident_id):
    rec = INCIDENTS.get(incident_id.strip().upper())
    if not rec:
        raise KeyError(f"No incident with ID {incident_id}")
    return rec


# ---- naive: everything the record has ----

@tool("Search the incident tracker. Returns the header line of every matching incident.", query=QUERY)
def search_incidents(query):
    hits = _search(query)
    return f"{len(hits)} matching incidents\n" + "\n".join(hits) if hits else "no matches"


@tool("Read one incident record in full: header, summary, timeline, log excerpt, impact and action items.",
      incident_id=ID)
def get_incident(incident_id):
    return render(_get(incident_id))


NAIVE = [search_incidents, get_incident]


# ---- trunc: the same tools, results cut at a fixed length ----

def truncated(t, limit=2000):
    fn = t.fn

    def cut(**kw):
        out = fn(**kw)
        return out if len(out) <= limit else out[:limit] + f"\n[... truncated {len(out) - limit:,} characters]"
    return type(t)(t.name, t.description, t.params, cut)


TRUNC = [truncated(t) for t in NAIVE]


# ---- sized: small results by design (key fields, pages, handles) ----

PAGE = 15


@tool("Search the incident tracker. Returns the header lines of matching incidents, "
      f"{PAGE} per page, with the total number of matches.",
      query=QUERY, page={"type": "integer", "description": "Page number, starting at 1"})
def search_incidents_paged(query, page):
    hits = _search(query)
    lo = (max(page, 1) - 1) * PAGE
    shown = hits[lo:lo + PAGE]
    if not hits:
        return "no matches"
    more = f"; call again with page={page + 1} for more" if lo + PAGE < len(hits) else ""
    return f"Matches {lo + 1}-{lo + len(shown)} of {len(hits)}{more}\n" + "\n".join(shown)


@tool("Read one incident's key fields: service, severity, times, follow-up incident, impact and root cause. "
      "The summary, timeline and log excerpt are not included; read the log with read_log.", incident_id=ID)
def get_incident_fields(incident_id):
    r = _get(incident_id)
    return "\n".join([
        f"{r['id']} | {r['service']} (owner: {r['team']}) | {r['severity']} | {r['title']}",
        f"Opened: {r['opened']} UTC  Resolved: {r['resolved']} UTC",
        f"Follow-up incident: {r['follow']}",
        f"Customer impact minutes: {r['impact_min']} | Robots affected: {r['robots']:,} | "
        f"Root cause category: {r['cause']}",
        f"Log excerpt: {len(r['log'])} lines, not shown; use read_log(\"{r['id']}\", start_line).",
    ])


@tool("Read 10 lines of an incident's log excerpt, starting at start_line (1-based).",
      incident_id=ID, start_line={"type": "integer", "description": "First line to show, starting at 1"})
def read_log(incident_id, start_line):
    log = _get(incident_id)["log"]
    lo = max(start_line, 1) - 1
    lines = [f"{i + 1:3d}  {log[i]}" for i in range(lo, min(lo + 10, len(log)))]
    return f"Lines {lo + 1}-{lo + len(lines)} of {len(log)}\n" + "\n".join(lines)


search_incidents_paged.name, get_incident_fields.name = "search_incidents", "get_incident"
SIZED = [search_incidents_paged, get_incident_fields, read_log]
TOOLSETS = {"naive": NAIVE, "trunc": TRUNC, "sized": SIZED, "edit": NAIVE, "compact": NAIVE, "memory": NAIVE}


# ---- grading ----

def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip().strip("*`. ") if m else None


def grade(task, text):
    a, exp = final_answer(text), task["answer"]
    if a is None:
        return False
    if isinstance(exp, int):
        nums = re.findall(r"\d[\d,]*", a)
        return bool(nums) and int(nums[0].replace(",", "")) == exp
    if exp.startswith("INC-"):
        ids = re.findall(r"INC-\d+", a.upper())
        return bool(ids) and ids[0] == exp
    # a category or an error code: the answer must name exactly one, the right one
    from corpus import CAUSES, CODES
    named = {c for c in CAUSES + CODES if re.search(rf"\b{re.escape(c.lower())}\b", a.lower())}
    return named == {exp}
