#!/bin/bash
# run each snippet in the task's buggy base (and nothing else)
cd /home/amit/.cache/vidyalai-scratch/a08
for id in $(venv/bin/python -c "import sys; sys.path.insert(0,'repro'); from snippets import S; print(' '.join(S))"); do
  b=tasks/$id/base; pp=$b; [ -d $b/src ] && pp=$b/src
  echo "=== $id"
  venv/bin/python -c "import sys; sys.path.insert(0,'repro'); from snippets import S; open('/home/amit/.cache/vidyalai-scratch/a08/repro/cur.py','w').write(S['$id'])"
  (cd repro && PYTHONPATH=$PWD/../$pp timeout 30 unshare -rn ../venv/bin/python cur.py 2>&1 | tail -6)
done
