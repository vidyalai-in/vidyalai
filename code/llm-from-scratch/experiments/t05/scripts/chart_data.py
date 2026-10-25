# Pull series out of the run logs as compact Python literals for the diagrams/t05-*.py generators.
import json, sys
R = "/home/amit/.cache/vidyalai-scratch/t05/runs/"
def load(name):
    rs = [json.loads(l) for l in open(R + name + "/log.jsonl")]
    tr = [(r["step"], round(r["loss"], 4), round(r["grad_norm"], 4), r["tokens"]) for r in rs if "loss" in r]
    va = [(r["step"], round(r["val_loss"], 4), r["tokens"]) for r in rs if "val_loss" in r]
    return tr, va
if __name__ == "__main__":
    for name in sys.argv[1:]:
        tr, va = load(name)
        print(name, "TRAIN", [t[1] for t in tr])
        print(name, "GN", [round(t[2], 3) for t in tr])
        print(name, "VAL", [(v[0], v[1]) for v in va])
