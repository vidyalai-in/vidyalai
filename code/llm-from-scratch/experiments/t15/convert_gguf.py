"""Run llama.cpp's convert_hf_to_gguf.py on our export, telling it which pre-tokenizer we use.

    python convert_gguf.py <hf_dir> --outtype f16 --outfile model-f16.gguf

The converter identifies a BPE pre-tokenizer by hashing the tokens of a test string. Ours is
unknown to it, but its split regex is exactly Qwen2's (GPT-4's pattern with single digits), so
we answer "qwen2" for our hash only, and leave llama.cpp's source untouched.
"""
import json
import runpy
import sys
from pathlib import Path

LLAMA = Path("/home/amit/.cache/vidyalai-scratch/t15/llama.cpp")
QWEN2_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
from conversion import base  # noqa: E402

stock = base.TextModel.get_vocab_base_pre


def get_vocab_base_pre(self, tokenizer):
    pre = json.load(open(self.dir_model / "tokenizer.json"))["pre_tokenizer"]
    if pre["pretokenizers"][0]["pattern"]["Regex"] == QWEN2_REGEX:   # Part 2's PATTERN
        return "qwen2"
    return stock(self, tokenizer)


base.TextModel.get_vocab_base_pre = get_vocab_base_pre
runpy.run_path(str(LLAMA / "convert_hf_to_gguf.py"), run_name="__main__")
