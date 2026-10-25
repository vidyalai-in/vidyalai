#!/bin/bash
# Part 6 main ladder and side studies: each configuration is its own short locked command.
D=/home/amit/.cache/vidyalai-scratch/t06
run() {  # run <name> <args...>
  name=$1; shift
  if grep -q '^RESULT' $D/logs/$name.out 2>/dev/null; then echo "skip $name"; return; fi
  export TORCHINDUCTOR_CACHE_DIR=$D/cache/$name TRITON_CACHE_DIR=$D/cache/$name/triton
  rm -rf $D/cache/$name
  $D/run.sh $name bench.py --name $name "$@"
  grep -o '"tok_s": [0-9.]*\|"peak_alloc_GiB": [0-9.]*\|"first_micro_s": [0-9.]*' $D/logs/$name.out | tr '\n' ' '; echo " <- $name"
}
for spec in "$@"; do eval "run $spec"; done
echo LADDER_DONE
