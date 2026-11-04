"""Experiment 4: a mini arena. Five Claude configurations answer the same arena prompts; Claude
Haiku 5.5 judges every pair in both orders; ratings come from lab/arena.py.
Usage: mini_arena.py pilot | gen N | judge N"""
import sys, json, re, time, itertools
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
P = "e07"
D = "/home/amit/.cache/vidyalai-scratch/e07"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
OFF = {"type": "disabled"}
CONFIGS = {
    "haiku-off": dict(model=HAIKU, thinking=OFF, max_tokens=3000),
    "haiku-off-markdown": dict(model=HAIKU, thinking=OFF, max_tokens=3000, system=
        "Answer with as much detail as possible. Format every answer with markdown: headers, bullet lists and bold key phrases."),
    "haiku-off-concise": dict(model=HAIKU, thinking=OFF, max_tokens=3000, system=
        "Be as concise as possible while still providing all the necessary information to answer the question."),
    "haiku-low": dict(model=HAIKU, thinking={"type": "adaptive"}, output_config={"effort": "low"}, max_tokens=3000),
    "sonnet-off": dict(model=SONNET, thinking={"type": "between_tools"}, max_tokens=2000),
}
JUDGE_SYS = ("You are a voter on a chatbot arena. A user sent the prompt below to two anonymous AI assistants. "
             "Decide which response you would prefer as that user. If they are equally good or equally bad, say TIE. "
             "Reply with exactly one word: A, B or TIE.")

def prompts(n):
    return [json.loads(l) for l in open(f"{D}/data/e07-arena-prompts.jsonl")][:n]

def gen_params(cfg, text):
    p = dict(CONFIGS[cfg]); p["messages"] = [{"role": "user", "content": text}]
    return p

def judge_params(prompt, ra, rb):
    msg = (f"<prompt>\n{prompt}\n</prompt>\n\n<response_A>\n{ra}\n</response_A>\n\n<response_B>\n{rb}\n</response_B>\n\n"
           "Which response is better? Reply with exactly one word: A, B or TIE.")
    return dict(model=HAIKU, thinking=OFF, max_tokens=8, system=JUDGE_SYS, messages=[{"role": "user", "content": msg}])

def text_of(msg):
    return "".join(c.text for c in msg.content if c.type == "text")

def parse(t):
    t = t.strip().strip("*.").upper()
    return t if t in ("A", "B", "TIE") else None

def run_batch(cl, reqs, tag):
    model_of = {r["custom_id"]: r["params"]["model"] for r in reqs}
    est = sum(ledger.estimate_cost(r["params"]["model"], len(json.dumps(r["params"])) // 2, r["params"]["max_tokens"]) for r in reqs) * 0.5
    print(f"{tag}: {len(reqs)} requests, worst-case ${est:.3f}", flush=True)
    ledger.guard(P, est)
    b = cl.messages.batches.create(requests=reqs)
    open(f"{D}/runs/{tag}.batch_id", "w").write(b.id)
    return collect(cl, b.id, model_of, tag)

def collect(cl, bid, model_of, tag):
    while cl.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(20)
    out = {}
    with open(f"{D}/runs/{tag}.jsonl", "w") as f:
        for it in cl.messages.batches.results(bid):
            rec = {"id": it.custom_id, "type": it.result.type}
            if it.result.type == "succeeded":
                m = it.result.message
                ledger.record(P, model_of[it.custom_id], m.usage, note=f"e07 {tag}", batch=True)
                rec.update(text=text_of(m), stop=m.stop_reason, usage=m.usage.model_dump(),
                           thought=any(c.type == "thinking" for c in m.content))
            f.write(json.dumps(rec) + "\n"); out[it.custom_id] = rec
    return out

if __name__ == "__main__":
    cl = ledger.client()
    mode = sys.argv[1]
    if mode == "pilot":
        log = open(f"{D}/runs/pilot.jsonl", "w")
        ps = prompts(2); ans = {}
        for i, p in enumerate(ps):
            for cfg in CONFIGS:
                m = ledger.create(cl, P, note=f"e07 pilot gen {cfg}", **gen_params(cfg, p["prompt"]))
                ans[(i, cfg)] = text_of(m)
                log.write(json.dumps({"i": i, "cfg": cfg, "stop": m.stop_reason, "usage": m.usage.model_dump(), "text": ans[(i, cfg)]}) + "\n")
                print(i, cfg, m.stop_reason, m.usage.input_tokens, m.usage.output_tokens, flush=True)
        for i, p in enumerate(ps):
            for x, y in [("haiku-off", "sonnet-off"), ("sonnet-off", "haiku-off")]:
                m = ledger.create(cl, P, note="e07 pilot judge", **judge_params(p["prompt"], ans[(i, x)], ans[(i, y)]))
                log.write(json.dumps({"i": i, "judge": [x, y], "usage": m.usage.model_dump(), "text": text_of(m)}) + "\n")
                print("judge", i, x, y, repr(text_of(m)), m.usage.input_tokens, m.usage.output_tokens, flush=True)
    elif mode == "gen":
        n = int(sys.argv[2]); ps = prompts(n)
        reqs = [{"custom_id": f"g{i}-{cfg}", "params": gen_params(cfg, p["prompt"])} for i, p in enumerate(ps) for cfg in CONFIGS]
        run_batch(cl, reqs, f"gen{n}")
    elif mode == "judge":
        n = int(sys.argv[2]); ps = prompts(n)
        gens = {r["id"]: r for r in map(json.loads, open(f"{D}/runs/gen{n}.jsonl"))}
        reqs = []
        for i, p in enumerate(ps):
            for x, y in itertools.permutations(CONFIGS, 2):
                ga, gb = gens.get(f"g{i}-{x}"), gens.get(f"g{i}-{y}")
                if ga and gb and ga["type"] == gb["type"] == "succeeded":
                    reqs.append({"custom_id": f"j{i}-{x}-vs-{y}", "params": judge_params(p["prompt"], ga["text"], gb["text"])})
        run_batch(cl, reqs, f"judge{n}")
    elif mode == "collect":
        tag = sys.argv[2]; bid = open(f"{D}/runs/{tag}.batch_id").read().strip()
        model_of = {}
        for it in cl.messages.batches.results(bid):
            model_of[it.custom_id] = SONNET if it.custom_id.endswith("sonnet-off") else HAIKU
        collect(cl, bid, model_of, tag)
