"""Build the Saltmere Islands world (a fictional knowledge base) and the 36-task set for Part 3.

Every answer is computed here from the generated records, so grading is exact.
Run: python world.py  -> writes world.json and tasks.json next to this file.
"""
import heapq
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(20261024)

ISLANDS = ["Brannock", "Elsk", "Toller", "Vey"]
TOWNS = {
    "Brannock": ["Port Wren", "Kettle Cove", "Ashby Fell", "Gorse Hill"],
    "Elsk": ["Hallow Quay", "Mirefield", "Stoneleigh", "Kettle Combe"],
    "Toller": ["Rookhaven", "Saltmarsh", "Penmoor", "Oxley Bay"],
    "Vey": ["Port Wrenna", "Lantern Point", "Fairholm", "Dunmere"],
}
FIRST = ["Ada", "Bram", "Cora", "Dmitri", "Edda", "Felix", "Greta", "Hugo", "Iris", "Jonas", "Kaia",
         "Lorcan", "Mira", "Nils", "Orla", "Piet", "Quinn", "Rosa", "Silas", "Tilde", "Ulric", "Vera",
         "Wim", "Xenia", "Yusuf", "Zelda", "Anouk", "Bastian", "Clio", "Dag"]
LAST = ["Abernethy", "Blackwood", "Calloway", "Dunstan", "Ellery", "Fairweather", "Gilchrist",
        "Hargreave", "Ingram", "Jessop", "Kincaid", "Lockhart", "Marchbank", "Nettleton", "Ormsby",
        "Pennington", "Quarrie", "Rutherford", "Sallow", "Thackeray", "Underhill", "Vickery",
        "Whitlock", "Yardley", "Ashcombe", "Brightwell", "Corrigan", "Delacourt", "Eastwick", "Frome"]
COMPANIES = ["Kestrel Mills", "Saltmere Canning", "Gannet Shipping", "Brightwater Salt",
             "Harrow Lanterns", "Oakum & Tar", "Tidewater Power", "Cobble Bank",
             "Fennel Breweries", "Weatherglass Instruments"]
FERRIES = ["MV Curlew", "MV Plover", "MV Dunlin", "MV Tern", "MV Skua", "MV Petrel", "MV Gannet II",
           "MV Fulmar"]

all_towns = [t for i in ISLANDS for t in TOWNS[i]]
island_of = {t: i for i in ISLANDS for t in TOWNS[i]}

# ---- people -------------------------------------------------------------------------------
names = [f"{f} {l}" for f, l in zip(FIRST, rng.sample(LAST, len(LAST)))]
people = {}
for n in names:
    people[n] = {"type": "person", "born": rng.randint(1941, 1994), "birthplace": rng.choice(all_towns)}

# ---- towns ------------------------------------------------------------------------------------
mayors = rng.sample(names, len(all_towns))
towns = {}
for t, m in zip(all_towns, mayors):
    towns[t] = {"type": "town", "island": island_of[t], "population": rng.randint(900, 48000),
                "area_km2": round(rng.uniform(3.0, 85.0), 1), "founded": rng.randint(1580, 1890),
                "mayor": m}
    people[m]["mayor_of"] = t

# ---- companies --------------------------------------------------------------------------------
non_mayors = [n for n in names if n not in mayors]
founders = rng.sample(non_mayors, len(COMPANIES))
companies = {}
for c, f in zip(COMPANIES, founders):
    born = people[f]["born"]
    founded = born + rng.randint(24, 48)
    if founded > 2024:
        founded = 2024 - rng.randint(0, 6)
    rev_thousand = rng.randint(300, 9800)
    companies[c] = {"type": "company", "founded": founded, "headquarters": rng.choice(all_towns),
                    "founder": f, "employees": rng.randint(12, 640), "revenue_thousand": rev_thousand}
    people[f]["founded"] = c

# ---- ferries ----------------------------------------------------------------------------------
ferries = {}
for fname in FERRIES:
    a, b = rng.sample(all_towns, 2)
    while island_of[a] == island_of[b]:
        a, b = rng.sample(all_towns, 2)
    ferries[fname] = {"type": "ferry", "route": [a, b], "operator": rng.choice(COMPANIES),
                      "departures_per_week": rng.randint(4, 30), "capacity": rng.choice(
                          [60, 80, 120, 150, 180, 220, 260, 300]), "crossing_min": rng.randint(25, 140)}

# ---- roads: 16 towns plus 14 crossroads, with many competing routes ---------------------------
CROSSROADS = ["Gallows Cross", "Mill Junction", "Heron Fork", "Tinker's Corner", "Barrow Gate",
              "Sheep Wash", "Coldharbour", "Wrack Lane End", "Five Ways", "Beacon Turn", "Smithy Cross",
              "Peat Bridge", "Old Toll", "Cairn Fork"]
places = all_towns + CROSSROADS
roads = {t: {} for t in places}
def add_road(a, b, km):
    roads[a][b] = km
    roads[b][a] = km
order = places[:]
rng.shuffle(order)
for i in range(1, len(order)):           # random spanning tree
    add_road(order[i], rng.choice(order[:i]), rng.randint(5, 30))
extra = 0
while extra < 22:                         # extra roads make several competing routes
    a, b = rng.sample(places, 2)
    if b not in roads[a]:
        add_road(a, b, rng.randint(12, 45))
        extra += 1


def dijkstra(src, banned=()):
    dist, pq = {src: 0}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in roads[u].items():
            if v in banned:
                continue
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def hops_and_trap(src, dst):
    """(fewest roads on any route, length of the shortest route that uses that few roads)."""
    hops, q = {src: 0}, [src]
    while q:
        nq = []
        for u in q:
            for v in roads[u]:
                if v not in hops:
                    hops[v] = hops[u] + 1
                    nq.append(v)
        q = nq
    best = {src: 0}
    for h in range(1, hops[dst] + 1):
        for v in [t for t in places if hops.get(t) == h]:
            best[v] = min(best[u] + roads[u][v] for u in roads[v] if hops.get(u) == h - 1)
    return hops[dst], best[dst]


def route_hops(src, dst, banned=()):
    """Number of roads on the shortest (by km) route."""
    dist, prev, pq = {src: 0}, {}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in roads[u].items():
            if v not in banned and d + w < dist.get(v, 1e9):
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    n, x = 0, dst
    while x != src:
        x, n = prev[x], n + 1
    return n


# ---- text records the lookup tool returns ------------------------------------------------------
def revenue_text(k):
    return f"{k / 1000:.3f} million crowns" if k >= 1000 else f"{k} thousand crowns"


records = {}
for i in ISLANDS:
    records[i] = (f"{i} (island). Towns: {', '.join(TOWNS[i])}. "
                  f"Total land area: {sum(towns[t]['area_km2'] for t in TOWNS[i]):.1f} km2.")
for t, r in towns.items():
    records[t] = (f"{t} (town) on the island of {r['island']}. Population: {r['population']:,}. "
                  f"Area: {r['area_km2']} km2. Founded: {r['founded']}. Mayor: {r['mayor']}.")
for n, r in people.items():
    s = f"{n} (person). Born: {r['born']} in {r['birthplace']}."
    if "mayor_of" in r:
        s += f" Mayor of {r['mayor_of']}."
    if "founded" in r:
        s += f" Founder of {r['founded']}."
    records[n] = s
for c, r in companies.items():
    records[c] = (f"{c} (company). Founded: {r['founded']} by {r['founder']}. "
                  f"Headquarters: {r['headquarters']}. Employees: {r['employees']}. "
                  f"Annual revenue: {revenue_text(r['revenue_thousand'])}.")
for x in CROSSROADS:
    records[x] = f"{x} (crossroads). A road junction with no town."
for f, r in ferries.items():
    records[f] = (f"{f} (ferry). Route: {r['route'][0]} - {r['route'][1]}. Operator: {r['operator']}. "
                  f"Departures per week: {r['departures_per_week']}. Passengers per departure: "
                  f"{r['capacity']}. Crossing time: {r['crossing_min']} minutes.")


# ---- tasks: 4 kinds x 9 -----------------------------------------------------------------------
tasks = []
def add(kind, q, answer, atype="int", **extra):
    tasks.append({"id": f"{kind}{sum(1 for t in tasks if t['kind'] == kind) + 1:02d}", "kind": kind,
                  "question": q, "answer": answer, "atype": atype, **extra})

def density(t):
    return towns[t]["population"] / towns[t]["area_km2"]

def unique_max(items, key):
    vals = sorted((key(x) for x in items), reverse=True)
    return vals[0] != vals[1]

def ferry_capacity(f):
    return ferries[f]["departures_per_week"] * ferries[f]["capacity"]

# chain: a superlative over a set, then a 3-4 hop chain, then arithmetic (9)
for prop, sign, label in [("crossing_min", 1, "the longest crossing time"),
                          ("departures_per_week", -1, "the fewest departures per week"),
                          ("capacity", -1, "the fewest passengers per departure")]:
    assert unique_max(FERRIES, lambda f: sign * ferries[f][prop]), prop
    f = max(FERRIES, key=lambda f: sign * ferries[f][prop])
    bp = people[companies[ferries[f]["operator"]]["founder"]]["birthplace"]
    add("chain", f"Find the ferry with {label}. The founder of the company that operates it was born in "
                 f"a town. What is that town's population density in people per km2, rounded to the "
                 f"nearest whole number?", round(density(bp)))
for i, f in [("Brannock", "MV Petrel"), ("Toller", "MV Tern"), ("Vey", "MV Skua")]:
    oldest = min(people[towns[t]["mayor"]]["born"] for t in TOWNS[i])
    fb = people[companies[ferries[f]["operator"]]["founder"]]["born"]
    add("chain", f"How many years older is the oldest mayor of a town on the island of {i} than the "
                 f"founder of the company that operates {f}? Use birth years; answer a negative number "
                 f"if the mayor is younger.", fb - oldest)
picked, used_isl = 0, set()
for c in COMPANIES:
    isl = island_of[people[companies[c]["founder"]]["birthplace"]]
    tot = sum(ferry_capacity(f) for f in FERRIES if island_of[ferries[f]["route"][0]] == isl
              or island_of[ferries[f]["route"][1]] == isl)
    nf = sum(1 for f in FERRIES if isl in (island_of[ferries[f]["route"][0]], island_of[ferries[f]["route"][1]]))
    if tot and nf >= 2 and picked < 3 and isl not in used_isl:
        picked += 1
        used_isl.add(isl)
        add("chain", f"The founder of {c} was born on one of the islands. What is the total weekly "
                     f"passenger capacity (departures per week times passengers per departure, summed) "
                     f"of all ferries with at least one end on that island?", tot)

# aggregate: enumerate a whole set, join, compute, rank or count (9)
rpe = {c: companies[c]["revenue_thousand"] / companies[c]["employees"] for c in COMPANIES}
ranked = sorted(COMPANIES, key=lambda c: -rpe[c])
for k, word in [(2, "second"), (3, "third"), (5, "fifth")]:
    add("aggregate", f"Rank all companies in the records by annual revenue per employee. Which company "
                     f"ranks {word}? Answer with the company name.", ranked[k - 1], "name")
for i in ["Brannock", "Elsk", "Vey"]:
    n = sum(1 for p in people.values() if island_of[p["birthplace"]] == i)
    add("aggregate", f"How many people in the records were born in a town on the island of {i}?", n)
for t in ["Penmoor", "Hallow Quay", "Lantern Point"]:
    y = people[towns[t]["mayor"]]["born"]
    tot = sum(towns[u]["population"] for u in all_towns if people[towns[u]["mayor"]]["born"] < y)
    add("aggregate", f"What is the total population of all towns (on every island) whose mayor was born "
                     f"in an earlier year than the mayor of {t}?", tot)

# route: shortest road distances where the fewest-roads route is a trap (9)
cands = []
for a, b in itertools.permutations(all_towns, 2):
    if a < b:
        d = dijkstra(a)[b]
        h, trap = hops_and_trap(a, b)
        if trap > d and route_hops(a, b) >= 4:
            cands.append((a, b, d, trap))
rng.shuffle(cands)
for a, b, d, trap in cands[:6]:
    add("route", f"Using the roads tool, find the length in km of the shortest road route from {a} to "
                 f"{b}. Routes may pass through towns and crossroads.", d, trap=trap)
banned_tasks = 0
for a, b, d, trap in cands[6:]:
    path_mid = [x for x in places if x not in (a, b) and dijkstra(a)[x] + dijkstra(x).get(b, 1e9) == d]
    for c in path_mid:
        d2 = dijkstra(a, banned={c}).get(b)
        if d2 and d2 > d and c in all_towns:
            add("route", f"Using the roads tool, find the length in km of the shortest road route from "
                         f"{a} to {b} that does not pass through {c}. Routes may pass through other towns "
                         f"and crossroads.", d2, unconstrained=d)
            banned_tasks += 1
            break
    if banned_tasks == 3:
        break

# puzzle: the next action depends on the last observation (9)
for _ in range(4):
    coin, delta = rng.randint(1, 12), rng.choice([1, -1])
    add("puzzle", "There are 12 coins numbered 1 to 12. All weigh the same except one, which is either "
                  "heavier or lighter; you don't know which. Use the weigh tool, which you may call at "
                  "most 3 times in total, to find the odd coin and whether it is heavier or lighter. "
                  "Answer like '7 heavier'.", f"{coin} {'heavier' if delta > 0 else 'lighter'}", "coin",
        scale={"n": 12, "max": 3, "odd": coin, "delta": delta})
for _ in range(5):
    code = "".join(rng.sample("0123456789", 4))
    add("puzzle", "A lock has a secret 4-digit code with four different digits (0-9). Each call to the "
                  "guess tool returns how many digits are right and in the right place (bulls) and how "
                  "many are right but in the wrong place (cows). You may guess at most 8 times. Find "
                  "the code and answer with its four digits.", code, "code", lock={"code": code, "max": 8})

world = {"records": records, "roads": roads}
(HERE / "world.json").write_text(json.dumps(world, indent=1))
(HERE / "tasks.json").write_text(json.dumps(tasks, indent=1))
if __name__ == "__main__":
    for t in tasks:
        print(t["id"], t["answer"], t.get("trap", t.get("unconstrained", "")), "|", t["question"][:100])
    print(len(tasks), "tasks;", len(records), "records;", sum(len(v) for v in roads.values()) // 2, "roads")
