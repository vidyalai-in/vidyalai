"""Download a FineWeb-Edu text sample and train the series' 32K byte-level BPE tokenizer.

    python train_tokenizer.py download          # ~400 MB sample + ~25 MB held-out
    python train_tokenizer.py train [vocab]     # writes tokenizer-32k.json (or tokenizer-<n>.json)

The sample streams from Hugging Face (streaming=True): 750 training documents and
50 held-out documents from each of FineWeb-Edu's Common Crawl snapshots, 2013-2025.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from tokenizer import PATTERN, SPECIAL_TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "data", "tokenizer-sample")
TRAIN_DOCS, HELDOUT_DOCS = 750, 50


def snapshot_names():
    from datasets import get_dataset_config_names
    names = get_dataset_config_names("HuggingFaceFW/fineweb-edu")
    return sorted(n for n in names if n.startswith("CC-MAIN-"))


def download_snapshot(name):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=name, split="train", streaming=True)
    docs = [row["text"] for row in islice(ds, TRAIN_DOCS + HELDOUT_DOCS)]
    return name, docs[:TRAIN_DOCS], docs[TRAIN_DOCS:]


def download():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    names = snapshot_names()
    with open(os.path.join(SAMPLE_DIR, "train.jsonl"), "w") as tr, \
         open(os.path.join(SAMPLE_DIR, "heldout.jsonl"), "w") as ho, \
         ThreadPoolExecutor(8) as pool:
        for name, train, heldout in pool.map(download_snapshot, names):
            for text in train:
                tr.write(json.dumps({"text": text, "dump": name}) + "\n")
            for text in heldout:
                ho.write(json.dumps({"text": text, "dump": name}) + "\n")
            print(name, len(train), len(heldout), flush=True)


def read_docs(split="train"):
    with open(os.path.join(SAMPLE_DIR, f"{split}.jsonl")) as f:
        for line in f:
            yield json.loads(line)["text"]


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    # Same regex as tokenizer.py, then map every byte to a printable character (GPT-2's trick).
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(vocab_size=32_768, docs=None, out=None):
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,                    # ids 0..7
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, ids 8..263
        show_progress=False,
    )
    t0 = time.perf_counter()
    tok.train_from_iterator(docs if docs is not None else read_docs("train"), trainer)
    print(f"trained vocab {tok.get_vocab_size()} in {time.perf_counter() - t0:.1f} s")
    out = out or os.path.join(HERE, f"tokenizer-{vocab_size // 1024}k.json")
    tok.save(out)
    return tok


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download()
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 32_768)
