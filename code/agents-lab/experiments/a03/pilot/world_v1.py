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

# ---- roads (a connected graph over all 16 towns) ----------------------------------------------
roads = {t: {} for t in all_towns}
def add_road(a, b, km):
    roads[a][b] = km
    roads[b][a] = km
order = all_towns[:]
rng.shuffle(order)
for i in range(1, len(order)):           # random spanning tree
    add_road(order[i], rng.choice(order[:i]), rng.randint(6, 40))
extra = 0
while extra < 12:                         # extra roads make several competing routes
    a, b = rng.sample(all_towns, 2)
    if b not in roads[a]:
        add_road(a, b, rng.randint(6, 40))
        extra += 1


def dijkstra(src):
    dist, pq = {src: 0}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in roads[u].items():
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def fewest_hops_route_km(src, dst):
    """Length of the shortest route among those with the fewest roads (the greedy trap answer)."""
    frontier, seen = [(src, 0)], {src: 0}
    best = None
    level = {src: 0}
    q = [src]
    hops = {src: 0}
    while q:
        nq = []
        for u in q:
            for v in roads[u]:
                if v not in hops:
                    hops[v] = hops[u] + 1
                    nq.append(v)
        q = nq
    # among paths with hops[dst] roads, the shortest
    best = {src: 0}
    for h in range(1, hops[dst] + 1):
        for v in [t for t in all_towns if hops.get(t) == h]:
            best[v] = min(best[u] + roads[u][v] for u in roads[v] if hops.get(u) == h - 1)
    return best[dst]


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
for f, r in ferries.items():
    records[f] = (f"{f} (ferry). Route: {r['route'][0]} - {r['route'][1]}. Operator: {r['operator']}. "
                  f"Departures per week: {r['departures_per_week']}. Passengers per departure: "
                  f"{r['capacity']}. Crossing time: {r['crossing_min']} minutes.")

# ---- tasks ----------------------------------------------------------------------------------
tasks = []
def add(kind, q, answer, atype="int"):
    tasks.append({"id": f"{kind}{sum(1 for t in tasks if t['kind'] == kind) + 1:02d}", "kind": kind,
                  "question": q, "answer": answer, "atype": atype})

def density(t):
    return towns[t]["population"] / towns[t]["area_km2"]

# chain: 3-4 hops plus arithmetic (12)
cs = COMPANIES[:]
for c in cs[:4]:
    hq = companies[c]["headquarters"]
    m = towns[hq]["mayor"]
    add("chain", f"How old was the current mayor of the town where {c} has its headquarters in the "
                 f"year {c} was founded? (Compute founding year minus birth year; it may be negative.)",
        companies[c]["founded"] - people[m]["born"])
for c in cs[4:8]:
    bp = people[companies[c]["founder"]]["birthplace"]
    add("chain", f"What is the population density, in people per km2 rounded to the nearest whole "
                 f"number, of the town where the founder of {c} was born?", round(density(bp)))
for f in FERRIES[:4]:
    op = ferries[f]["operator"]
    hq = companies[op]["headquarters"]
    add("chain", f"The company that operates {f} has its headquarters in a town. How many years before "
                 f"that company was founded was that town founded?",
        companies[op]["founded"] - towns[hq]["founded"])

# aggregate: enumerate a set, compute, filter or compare (12)
for i in ISLANDS:
    best = max(TOWNS[i], key=density)
    add("aggregate", f"Which town on the island of {i} has the highest population density "
                     f"(population divided by area)? Answer with the town name.", best, "name")
for i in ISLANDS:
    tot = sum(r["revenue_thousand"] for r in companies.values() if island_of[r["headquarters"]] == i)
    add("aggregate", f"What is the combined annual revenue, in thousand crowns, of all companies "
                     f"headquartered on the island of {i}? Answer 0 if there are none.", tot)
for i, year in zip(ISLANDS, [1965, 1970, 1975, 1960]):
    tot = sum(towns[t]["population"] for t in TOWNS[i] if people[towns[t]["mayor"]]["born"] < year)
    add("aggregate", f"What is the total population of the towns on the island of {i} whose mayor was "
                     f"born before {year}? Answer 0 if there are none.", tot)

# puzzles: the next action depends on the last observation (12)
pairs = []
for a, b in itertools.permutations(all_towns, 2):
    d = dijkstra(a)[b]
    trap = fewest_hops_route_km(a, b)
    if trap > d and len(roads[a]) >= 2:
        pairs.append((a, b, d, trap))
rng.shuffle(pairs)
used = set()
for a, b, d, trap in pairs:
    if (a, b) in used or (b, a) in used or a in {x for x, _ in used}:
        continue
    used.add((a, b))
    add("puzzle", f"Using only the roads tool, find the length in km of the shortest road route from "
                  f"{a} to {b}.", d)
    tasks[-1]["trap"] = trap
    if sum(1 for t in tasks if t["kind"] == "puzzle") == 4:
        break
for n, w in [(9, 2), (8, 2), (27, 3), (12, 3)]:
    coin = rng.randint(1, n)
    add("puzzle", f"There are {n} coins numbered 1 to {n}. All weigh the same except one, which is "
                  f"heavier. Use the weigh tool, which you may call at most {w} times in total, to find "
                  f"the heavy coin. Answer with its number.", coin)
    tasks[-1]["scale"] = {"n": n, "max": w, "heavy": coin}
for _ in range(4):
    code = "".join(rng.sample("0123456789", 3))
    add("puzzle", "A lock has a secret 3-digit code with three different digits (0-9). Each call to the "
                  "guess tool returns how many digits are right and in the right place (bulls) and how "
                  "many are right but in the wrong place (cows). You may guess at most 10 times. Find "
                  "the code and answer with its three digits.", code, "code")
    tasks[-1]["lock"] = {"code": code, "max": 10}

world = {"records": records, "roads": roads}
(HERE / "world.json").write_text(json.dumps(world, indent=1))
(HERE / "tasks.json").write_text(json.dumps(tasks, indent=1))
if __name__ == "__main__":
    for t in tasks:
        print(t["id"], t["answer"], t.get("trap", ""), "|", t["question"][:110])
    print(len(records), "records;", sum(len(v) for v in roads.values()) // 2, "roads")
