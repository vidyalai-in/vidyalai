#!/bin/bash
# llama-perplexity on the held-out slice for every GGUF (CPU, 8 threads), then KL divergence
# of each quantized file against the F32 file's log-probabilities on the first 40 chunks.
T=/home/amit/.cache/vidyalai-scratch/t15
BIN=$T/llama.cpp/build/bin
mkdir -p $T/post/ppl
for m in base88 chat88; do
  for t in f32 f16 q8_0 q4_k_m; do
    /usr/bin/time -f "%e s wall" $BIN/llama-perplexity -m $T/gguf/$m-$t.gguf -f $T/post/heldout.txt \
      --no-escape -c 1024 -b 1024 -t 8 > $T/post/ppl/$m-$t.log 2>&1
    grep -E 'Final estimate|s wall' $T/post/ppl/$m-$t.log | tr '\n' ' '; echo " $m $t"
  done
  $BIN/llama-perplexity -m $T/gguf/$m-f32.gguf -f $T/post/heldout.txt --no-escape -c 1024 -b 1024 -t 8 \
    --chunks 40 --kl-divergence-base $T/post/kld-$m-f32.bin > $T/post/ppl/$m-kldbase.log 2>&1
  for t in f16 q8_0 q4_k_m; do
    $BIN/llama-perplexity -m $T/gguf/$m-$t.gguf --no-escape -c 1024 -b 1024 -t 8 --chunks 40 \
      --kl-divergence-base $T/post/kld-$m-f32.bin --kl-divergence > $T/post/ppl/$m-$t-kld.log 2>&1
  done
  rm -f $T/post/kld-$m-f32.bin
done
