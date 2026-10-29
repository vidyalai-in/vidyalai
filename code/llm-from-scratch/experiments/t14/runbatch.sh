#!/bin/bash
# usage: runbatch.sh <batch file>: one bench.py run per line (tag, host memory cap, args), each under the GPU lock
S=/home/amit/.cache/vidyalai-scratch/t14/scripts; L=/home/amit/.cache/vidyalai-scratch/t14/logs
while read -r tag cap args; do
  [ -z "$tag" ] && continue
  [ -f /home/amit/.cache/vidyalai-scratch/t14/runs/$tag.json ] && continue      # done already: resume-safe
  timeout 900 $S/run.sh $tag $cap $S/bench.py --tag $tag --compile $args > $L/$tag.out 2>&1
  echo "$tag $(tail -1 $L/$tag.out | cut -c1-200)" >> $L/batch.log
done < "$1"
