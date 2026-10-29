#!/bin/bash
# HF export -> GGUF F32 and F16 (convert_hf_to_gguf.py via our wrapper) -> Q8_0 and Q4_K_M (llama-quantize).
set -e
T=/home/amit/.cache/vidyalai-scratch/t15
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
BIN=$T/llama.cpp/build/bin
mkdir -p $T/gguf
for m in base88 chat88; do
  for t in f32 f16; do
    $PY $T/scripts/convert_gguf.py $T/export/$m --outtype $t --outfile $T/gguf/$m-$t.gguf > $T/gguf/$m-$t.log 2>&1
  done
  for q in Q8_0 Q4_K_M; do
    $BIN/llama-quantize $T/gguf/$m-f32.gguf $T/gguf/$m-${q,,}.gguf $q > $T/gguf/$m-${q,,}.log 2>&1
  done
done
ls -l $T/gguf/*.gguf
