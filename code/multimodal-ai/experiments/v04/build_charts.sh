#!/bin/bash
# Refresh results.json, paste the data blocks into the two chart generators, and render them.
set -e
cd /home/amit/.cache/vidyalai-scratch/v04
systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python analyze.py
python3 make_chart_data.py > chart_data.txt
cd /home/amit/AlpsCode/ai/vidyalai
python3 - <<'PY'
import re
d = dict(l.split(" = ", 1) for l in open("/home/amit/.cache/vidyalai-scratch/v04/chart_data.txt").read().splitlines())
for f, k in (("diagrams/v04-accuracy-by-size.py", "ACC"), ("diagrams/v04-text-height.py", "HEIGHT")):
    s = open(f).read()
    s = re.sub(rf"^{k} = .*$", f"{k} = {d[k]}", s, count=1, flags=re.M)
    open(f, "w").write(s)
PY
for n in v04-accuracy-by-size v04-text-height; do
  python3 diagrams/$n.py public/diagrams/$n
  node -e "require('sharp')('public/diagrams/$n.svg',{density:144}).png().toFile('public/diagrams/$n.png')"
done
ls -la public/diagrams/v04-*
