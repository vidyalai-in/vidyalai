import sys, json, regex as re
from collections import Counter
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/t02/scripts")
from tokenizer import PATTERN
from slice import slice_docs
docs, n = slice_docs("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/train.jsonl", 10)
c = Counter(p for d in docs for p in re.findall(PATTERN, d)); print("10MB distinct", len(c), "total", sum(c.values()), "bytes in distinct", sum(len(k.encode()) for k in c))
c = Counter()
for l in open("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/train.jsonl"):
    c.update(re.findall(PATTERN, json.loads(l)["text"]))
print("406MB distinct", len(c), "total", sum(c.values()), "bytes in distinct", sum(len(k.encode()) for k in c))
