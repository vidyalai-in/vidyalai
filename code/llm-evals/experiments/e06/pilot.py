"""Pilot: a few single calls of each judge type, to check parsing and cost per item."""
import json
import sys

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import grading as G
import ledger

D = "/home/amit/.cache/vidyalai-scratch/e06/"
cl = ledger.client()
se = json.load(open(D + "data/summeval.json"))
mt = json.load(open(D + "data/mtbench.json"))
log = open(D + "runs/pilot.jsonl", "a")


def go(cid, p, note):
    r = ledger.create(cl, "e06", note=note, **p)
    text = "".join(c.text for c in r.content if c.type == "text")
    log.write(json.dumps({"id": cid, "params": p, "text": text, "stop": r.stop_reason,
                          "usage": r.usage.model_dump()}) + "\n")
    print(cid, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, "|", text[-160:].replace("\n", " "))


for i in range(3):
    d = se[i * 7]
    go(f"rub-{i}", G.rubric_request(d["article"], d["summaries"][i], anchored=True), "e06 pilot rubric")
go("bare-0", G.rubric_request(se[0]["article"], se[0]["summaries"][0], anchored=False), "e06 pilot bare")
for i in range(3):
    p = mt[i * 97]
    go(f"pair-{i}", G.pair_request(p["question"], p["answer_1"], p["answer_2"]), "e06 pilot pair")
p = [x for x in mt if x["reference"]][5]
go("ref-0", G.pair_request(p["question"], p["answer_1"], p["answer_2"], reference=p["reference"]), "e06 pilot ref")
go("single-0", G.single_request(mt[0]["question"], mt[0]["answer_1"]), "e06 pilot single")
go("sonnet-pair-0", G.pair_request(mt[0]["question"], mt[0]["answer_1"], mt[0]["answer_2"], model="claude-sonnet-5-5"),
   "e06 pilot sonnet pair")
