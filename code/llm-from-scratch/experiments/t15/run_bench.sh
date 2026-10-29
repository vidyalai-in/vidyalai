#!/bin/bash
# llama-bench on the laptop CPU: 512-token prompt processing and 128-token generation, 5 repetitions,
# at 8 threads (one per physical core) and 16 (all hardware threads).
T=/home/amit/.cache/vidyalai-scratch/t15
M=$(ls $T/gguf/chat88-{f32,f16,q8_0,q4_k_m}.gguf | paste -sd,)
for t in 8 16; do
  $T/llama.cpp/build/bin/llama-bench -m $M -p 512 -n 128 -t $t -r 5 -o json > $T/post/bench_t$t.json 2> $T/post/bench_t$t.err
done
grep -h model_type $T/post/bench_t8.json | head -1
