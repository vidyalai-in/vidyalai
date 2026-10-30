# Harder city facts for the sycophancy probe: GeoNames (CC BY 4.0) cities of 100,000-500,000 people,
# same filters as Part 3 (unique name, ASCII, populated place, name doesn't contain the country),
# at most 20 per country, countries with >= 3; one false country drawn by frequency, as in Part 3.
import collections, json, random
G = "/home/amit/.cache/vidyalai-scratch/interp/data/p03/geonames"
OUT = "/home/amit/.cache/vidyalai-scratch/p11/data/hard_cities.json"
names = {}
for line in open(f"{G}/countryInfo.txt", encoding="utf-8"):
    if not line.startswith("#"):
        f = line.split("\t"); names[f[0]] = f[4]
rows = [l.rstrip("\n").split("\t") for l in open(f"{G}/cities15000.txt", encoding="utf-8")]
cnt = collections.Counter(r[1] for r in rows)
part3 = {d["city"] for d in json.load(open("/home/amit/.cache/vidyalai-scratch/p03/runs/truth_data.json"))}
cand = [r for r in rows if 100_000 <= int(r[14]) < 500_000 and r[1] == r[2] and cnt[r[1]] == 1
        and r[1].isascii() and r[7].startswith("PPL") and names[r[8]] not in r[1] and r[1] not in part3]
by = collections.defaultdict(list)
for r in cand: by[r[8]].append(r)
cities = []
random.seed(0)
for cc, rs in sorted(by.items()):
    if len(rs) < 3: continue
    random.shuffle(rs)
    cities += [(r[1], names[cc]) for r in rs[:20]]
freq = collections.Counter(c for _, c in cities)
cs, ws = zip(*freq.items())
out = []
for city, country in cities:
    while True:
        fc = random.choices(cs, ws)[0]
        if fc != country: break
    out.append({"city": city, "true": country, "false": fc})
print(len(out), "cities", len(freq), "countries", freq.most_common(5))
json.dump(out, open(OUT, "w"))
