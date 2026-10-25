"""A readable byte-level BPE tokenizer in pure Python (Build an LLM From Scratch, Part 2).

The series trains its production tokenizer with Hugging Face `tokenizers` (train_tokenizer.py),
using the same PATTERN and SPECIAL_TOKENS; load it with load_production().
"""
import os
from collections import Counter

import regex as re

# GPT-4's pre-tokenization regex (as used by Llama 3), with numbers split into single digits.
PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Document boundary, chat roles for Part 10, and three spares.
SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"]


def pair_counts(chunks):
    """Count adjacent id pairs; each chunk is weighted by how often it occurs."""
    counts = Counter()
    for ids, n in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += n
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train_bpe(texts, n_merges, verbose=False):
    """Learn `n_merges` byte-pair merges. Returns merges {(id, id): new_id} and vocab {id: bytes}."""
    # 1. Pre-tokenize, and count identical pieces once ("the" may occur a million times).
    chunks = Counter()
    for text in texts:
        for piece in re.findall(PATTERN, text):
            chunks[tuple(piece.encode("utf-8"))] += 1
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    # 2. Merge the most frequent pair, n_merges times.
    for new_id in range(256, 256 + n_merges):
        counts = pair_counts(chunks)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = {merge(ids, pair, new_id): n for ids, n in chunks.items()}
        if verbose:
            print(new_id, vocab[new_id], counts[pair])
    return merges, vocab


class BPETokenizer:
    def __init__(self):
        self.merges = {}                                   # (id, id) -> new id, in learned order
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> the bytes it stands for
        self.special = {}                                  # special token string -> id
        self.cache = {}

    def train(self, texts, vocab_size, verbose=False):
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        self.merges, self.vocab = train_bpe(texts, n_merges, verbose)
        # Special tokens go last; they never come out of merges.
        for tok in SPECIAL_TOKENS:
            self.special[tok] = len(self.vocab)
            self.vocab[len(self.vocab)] = tok.encode("utf-8")
        self.cache = {}

    def encode_chunk(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        ids = tuple(piece.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, exactly as training did.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self.cache[piece] = ids
        return ids

    def encode(self, text, allow_special=True):
        ids = []
        parts = [text]
        if allow_special and self.special:
            parts = re.split("(" + "|".join(map(re.escape, self.special)) + ")", text)
        for part in parts:
            if part in self.special and allow_special:
                ids.append(self.special[part])
            else:
                for piece in re.findall(PATTERN, part):
                    ids.extend(self.encode_chunk(piece))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def load_production(path=None):
    """The series' 32,768-token tokenizer (Hugging Face `tokenizers` format)."""
    from tokenizers import Tokenizer
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer-32k.json")
    return Tokenizer.from_file(path)


def _gpt2_byte_chars():
    """GPT-2's map from printable characters back to bytes, as used in `tokenizers` ByteLevel files."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def python_from_production(path=None):
    """A pure-Python BPETokenizer holding the production merges (ids differ: bytes are 0-255 here)."""
    import json
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer-32k.json")
    to_byte = _gpt2_byte_chars()
    tok, by_bytes = BPETokenizer(), {bytes([i]): i for i in range(256)}
    for k, m in enumerate(json.load(open(path))["model"]["merges"]):
        a, b = (bytes(to_byte[c] for c in s) for s in (m.split(" ") if isinstance(m, str) else m))
        tok.merges[(by_bytes[a], by_bytes[b])] = 256 + k
        tok.vocab[256 + k] = a + b
        by_bytes[a + b] = 256 + k
    for t in SPECIAL_TOKENS:
        tok.special[t] = len(tok.vocab)
        tok.vocab[len(tok.vocab)] = t.encode("utf-8")
    return tok
