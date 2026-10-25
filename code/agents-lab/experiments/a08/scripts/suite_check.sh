#!/bin/bash
# Run each repo's test suite at HEAD (no network, 2 GB cap, 20 s per test, 240 s per repo).
cd /home/amit/.cache/vidyalai-scratch/a08/repos
for d in "$@"; do
  pp=$PWD/$d; [ -d $d/src ] && pp=$PWD/$d/src
  t0=$(date +%s)
  out=$(cd $d && PYTHONPATH=$pp timeout 240 systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 unshare -rn ../../venv/bin/python -m pytest -q -p no:cacheprovider --timeout=20 tests 2>&1 | tail -1)
  echo "$d [$(( $(date +%s)-t0 ))s] $out"
done
