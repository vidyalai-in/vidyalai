# Build true/false city statements from GeoNames (CC BY 4.0), following Marks & Tegmark (2023, App. H):
# cities with population > 500,000 whose name no other GeoNames city shares, not city-states;
# one true statement and one false one per city, the false country sampled with probability equal
# to its frequency among the true statements. Two differences: we cap each country at its 40 most
# populous cities, and keep countries with at least 3 qualifying cities.
import collections, json, random
G = "/home/amit/.cache/vidyalai-scratch/interp/data/p03/geonames"
OUT = "/home/amit/.cache/vidyalai-scratch/p03/runs/truth_data.json"
names = {}
for line in open(f"{G}/countryInfo.txt", encoding="utf-8"):
    if line.startswith("#"):
        continue
    f = line.split("\t")
    names[f[0]] = f[4]
rows = [l.rstrip("\n").split("\t") for l in open(f"{G}/cities15000.txt", encoding="utf-8")]
name_count = collections.Counter(r[1] for r in rows)
CITY_STATES = {"SG", "HK", "MO", "VA", "MC", "KW", "BH", "QA", "DJ"}
cand = [r for r in rows if int(r[14]) > 500_000 and r[1] == r[2] and name_count[r[1]] == 1
        and r[8] not in CITY_STATES and r[1].isascii() and names[r[8]] not in r[1]
        and r[7].startswith("PPL")]
by_c = collections.defaultdict(list)
for r in cand:
    by_c[r[8]].append(r)
cities = []
for cc, rs in by_c.items():
    if len(rs) < 3:
        continue
    rs.sort(key=lambda r: -int(r[14]))
    cities += [(r[1], names[cc], int(r[14])) for r in rs[:40]]
random.seed(0)
freq = collections.Counter(c for _, c, _ in cities)
countries, weights = zip(*freq.items())
items = []
for city, country, pop in cities:
    while True:
        fc = random.choices(countries, weights)[0]
        if fc != country:
            break
    items.append({"city": city, "country": country, "true_country": country, "label": 1, "pop": pop})
    items.append({"city": city, "country": fc, "true_country": country, "label": 0, "pop": pop})
print("cities", len(cities), "countries", len(freq), "statements", len(items))
print(freq.most_common(8))
json.dump(items, open(OUT, "w"))
