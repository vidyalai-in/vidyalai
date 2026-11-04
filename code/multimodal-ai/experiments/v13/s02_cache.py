"""Exp 1b: prompt caching with images on claude-haiku-5-5 (thinking disabled), streamed.
Three arms, interleaved in rounds (order shuffled within each round), one new question per round:
  plain  the same page image(s), no cache_control           -> every call reads the image fresh
  write  cache_control on the last image, but the first page carries a unique footer tag,
         so every call writes a new cache entry               -> cache miss + write
  hit    cache_control on the last image, identical image(s) -> cache read (primed once beforehand)
Sizes: S = page 1 at 1200 x 800 (1 image), L = page 1 at 2352 x 1568, XL = pages 1-8 at 2352 x 1568.
Usage: s02_cache.py SIZE ROUNDS [TAGPREFIX]"""
import json, random, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ledger, serving as S

D = "/home/amit/.cache/vidyalai-scratch/v13"
MODEL = "claude-haiku-5-5"
SYS = "Answer questions about the attached report pages. Reply with the number only."
size, rounds = sys.argv[1], int(sys.argv[2])
tagp = sys.argv[3] if len(sys.argv) > 3 else "r"
dims = {"S": (1200, 800, 1), "L": (2352, 1568, 1), "XL": (2352, 1568, 8)}[size]
w, h, npages = dims
pages = [S.report_page(i, w, h) for i in range(npages)]
qs = S.page_questions(range(npages), rounds + 1, rng_seed={"S": 1, "L": 2, "XL": 3}[size], extra=(npages == 1))
est_in = {"S": 1310, "L": 4767, "XL": 37716}[size] + 50
est = ledger.estimate_cost(MODEL, est_in, 40)
cl = ledger.client()
out = open(f"{D}/runs/cache_{size}{"_pilot" if tagp == "p" else ""}.jsonl", "a")


BLOCKS = [S.image_block(im) for im in pages]   # encode the fixed pages once


def content(arm, q, tag=None):
    blocks = [dict(b) for b in BLOCKS]
    if tag is not None:
        blocks[0] = S.image_block(S.report_page(0, w, h, tag=tag))
    if arm != "plain":
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks + [{"type": "text", "text": q}]


def call(arm, r, q, a, tag=None):
    rec = {"size": size, "arm": arm, "round": r, "q": q, "answer": a, "wall": time.time(), "tag": tag}
    try:
        res = S.stream_call(cl, ledger, "v13", MODEL, content(arm, q, tag), SYS, 40,
                            note=f"v13 cache {size} {arm} r{r}", est=est)
        rec.update(res)
        rec["ok"] = res["text"].strip().rstrip(".") == a
    except ledger.BudgetExceeded as e:
        rec["error"] = "budget: " + str(e)
        out.write(json.dumps(rec) + "\n"); out.flush()
        raise
    except Exception as e:  # network errors are logged and the arm continues
        rec["error"] = repr(e)
    out.write(json.dumps(rec) + "\n"); out.flush()
    u = rec.get("usage", {})
    print(size, arm, r, rec.get("ttft") and round(rec["ttft"], 3), u.get("input_tokens"),
          u.get("cache_creation_input_tokens"), u.get("cache_read_input_tokens"), u.get("output_tokens"),
          repr(rec.get("text")), rec.get("ok"), rec.get("error", ""), flush=True)


rng = random.Random(7)
q0, a0 = qs[0]
call("hit", -1, q0, a0)                       # prime: writes the cache entry the hit arm reads
for r in range(rounds):
    q, a = qs[r + 1]
    arms = ["plain", "write", "hit"]
    rng.shuffle(arms)
    for arm in arms:
        call(arm, r, q, a, tag=f"{tagp}{size}{r:03d}{random.randrange(10**6):06d}" if arm == "write" else None)
