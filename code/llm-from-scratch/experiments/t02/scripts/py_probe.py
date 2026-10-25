import sys, time; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/t02/scripts")
from slice import slice_docs
from tokenizer import BPETokenizer
docs, n = slice_docs("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/train.jsonl", float(sys.argv[1]))
t = BPETokenizer(); t0 = time.perf_counter(); t.train(docs, int(sys.argv[2])); print(n, "bytes", time.perf_counter()-t0, "s")
