#!/bin/bash
# Part 8: the scaling sweep. Run the whole script under the GPU lock (it must not take the lock itself):
#   flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash run_sweep.sh [extra train.py args]
# Trains each run below in order, smallest first, into $ROOT/<name>/. A run whose log.jsonl already has
# an "end" event is skipped; a crashed run is resumed from its ckpt.pt (up to $TRIES tries, $RETRY_SLEEP
# seconds apart); if a run still fails, the sweep stops. nvidia-smi logs every second to logs/<name>.smi.csv.
#   ONLY="s11 s15"  run only these names          ROOT=<dir>  where run directories go (default runs/)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
D=/home/amit/.cache/vidyalai-scratch/t08
ROOT=${ROOT:-$D/runs}; LOGS=${LOGS:-$D/logs}; TRIES=${TRIES:-4}; RETRY_SLEEP=${RETRY_SLEEP:-120}
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/fromscratch
mkdir -p $ROOT $LOGS

# name     model accum steps warmup peak_lr  val_every ckpt_every   (micro-batch 16 x 1,024 tokens)
# Main sweep: 20 tokens per (total) parameter. Tokens per step = 16,384 x accum, about 524,288 x N / 88.1M
# rounded to a power of two, so every size takes about as many steps as Part 7's 88M run (3,360).
# Peak learning rate 0.6 / d_model; warm-up 4.5% of the steps (Part 7: 150 of 3,360); cosine to 10%.
# IsoFLOP extras (iso_*): the same per-size settings, trained for the compute of s15's main run
# (6 x 14,749,344 x 294,977,536 FLOPs): s11 at 36.4 tokens per parameter, s22 at 9.0.
RUNS="
s11      s11    4   3338  150  0.0025    83  167
s15      s15    4   4501  203  0.00208  113  225
s22      s22    8   3361  151  0.00156   84  168
s33      s33   16   2513  113  0.00125   63  126
s47      s47   16   3601  162  0.00104   90  180
iso_s11  s11    4   6069  273  0.0025   152  303
iso_s22  s22    8   1507   68  0.00156   38   75
"
SMI_Q=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used,clocks_event_reasons.active

ended() { grep -q '"event": "end"' "$1/log.jsonl" 2>/dev/null; }

echo "=== sweep start $(date -Is) ROOT=$ROOT args: $*" >> $LOGS/sweep.out
while read -r name model accum steps warmup lr val_every ckpt_every; do
  [ -z "$name" ] && continue
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  OUT=$ROOT/$name
  if ended $OUT; then echo "skip $name (ended) $(date -Is)" >> $LOGS/sweep.out; continue; fi
  mkdir -p $OUT
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=$SMI_Q --format=csv,noheader -lms 1000 >> $LOGS/$name.smi.csv < /dev/null &
  SMI=$!
  for try in $(seq 1 $TRIES); do
    echo "=== $name try $try $(date -Is)" | tee -a $LOGS/sweep.out >> $LOGS/$name.out
    systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
      $PY train.py --model $model --out $OUT --micro-batch 16 --accum $accum \
        --steps $steps --warmup $warmup --lr $lr --val-every $val_every \
        --ckpt-every $ckpt_every --keep-ckpts --compile --fused --resume "$@" >> $LOGS/$name.out 2>&1 < /dev/null
    echo "=== $name exit $? $(date -Is)" | tee -a $LOGS/sweep.out >> $LOGS/$name.out
    ended $OUT && break
    [ $try -lt $TRIES ] && sleep $RETRY_SLEEP
  done
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  if ! ended $OUT; then echo "=== sweep stopped: $name failed $TRIES tries $(date -Is)" >> $LOGS/sweep.out; exit 1; fi
  grep -q '"event": "diverged"' $OUT/log.jsonl && echo "WARNING $name diverged" >> $LOGS/sweep.out
done <<< "$RUNS"
echo "=== sweep end $(date -Is)" >> $LOGS/sweep.out
