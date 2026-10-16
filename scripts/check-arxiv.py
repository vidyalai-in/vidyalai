"""Checks that each arXiv link's text matches the paper title arXiv reports for that ID."""
import re, glob, urllib.request, xml.etree.ElementTree as ET, difflib, time

pairs = {}
for f in sorted(glob.glob('src/content/blog/*.md')):
    for text, aid in re.findall(r'\[([^\]]+)\]\(https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', open(f).read()):
        pairs.setdefault(aid, set()).add((text, f.split('/')[-1][:2]))

ids = sorted(pairs)
titles = {}
for i in range(0, len(ids), 40):
    url = 'http://export.arxiv.org/api/query?max_results=100&id_list=' + ','.join(ids[i:i + 40])
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    for e in root.findall('a:entry', ns):
        aid = re.search(r'(\d{4}\.\d{4,5})', e.find('a:id', ns).text).group(1)
        titles[aid] = ' '.join(e.find('a:title', ns).text.split())
    time.sleep(3)

norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower())
print(f'{len(ids)} arXiv IDs')
for aid in ids:
    real = titles.get(aid)
    for text, post in pairs[aid]:
        cited = re.sub(r'\s*\([^)]*\d{4}[^)]*\)\s*$', '', text)  # drop "(Author, 2024)"
        score = difflib.SequenceMatcher(None, norm(cited), norm(real or '')).ratio()
        contained = norm(cited) in norm(real or '') or norm(real or '').startswith(norm(cited)[:25])
        if not real or (score < 0.6 and not contained):
            print(f'CHECK {aid} [{post}] cited="{text}" | arXiv="{real}" ({score:.2f})')
