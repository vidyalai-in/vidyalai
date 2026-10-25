from collections import Counter
import regex as re
from tokenizer import PATTERN    # the regex above

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
