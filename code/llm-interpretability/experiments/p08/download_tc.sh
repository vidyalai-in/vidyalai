#!/bin/bash
# Download Gemma Scope 2B PT transcoders (16k width, the lowest-L0 set used by circuit-tracer's
# mntss/gemma-scope-transcoders config). CC-BY-4.0, not gated. No token used.
cd /home/amit/.cache/vidyalai-scratch/interp/data/p08/gs-transcoders
while read p; do
  out=$(echo $p | tr '/' '_').npz
  [ -s "$out" ] && [ $(stat -c %s "$out") -eq 302131416 ] && continue
  curl -sL --retry 3 -o "$out.part" "https://huggingface.co/google/gemma-scope-2b-pt-transcoders/resolve/main/$p/params.npz" && mv "$out.part" "$out"
  echo "$(date +%T) $out $(stat -c %s $out)"
done < layers.txt
echo DONE
