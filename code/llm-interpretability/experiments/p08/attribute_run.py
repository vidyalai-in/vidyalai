# Build attribution graphs with circuit-tracer on Gemma 2 2B + Gemma Scope transcoders (lowest-L0,
# 16k features per layer), report graph size, replacement/completeness scores, time and memory.
import sys, json, time, resource, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.attribution import load_gemma_tl, load_transcoders, graph_summary, top_nodes
from circuit_tracer import attribute

PROMPTS = {
    "dallas": "Fact: the capital of the state containing Dallas is",
    "brandenburg": "The Brandenburg Gate is located in the city of",
    "ioi": "When Mary and John went to the store, John gave a drink to",
}
which = sys.argv[1:] or list(PROMPTS)
BATCH = 64

t0 = time.time()
tcs = load_transcoders(lazy=True)
model = load_gemma_tl(tcs, n_ctx=64)
print(f"loaded in {time.time()-t0:.1f}s, GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

for name in which:
    prompt = PROMPTS[name]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    g = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95,
                  batch_size=BATCH, max_feature_nodes=8192, offload=None, verbose=False)
    secs = time.time() - t
    g.to_pt(f"/home/amit/.cache/vidyalai-scratch/p08/runs/graph_{name}.pt")
    s = graph_summary(g)
    s.update(name=name, seconds=secs, batch_size=BATCH,
             gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
             host_rss_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
             logit_targets=[model.tokenizer.decode([int(t)]) if not isinstance(t, str) else t
                            for t in g.logit_token_ids.tolist()],
             logit_probs=[round(p, 4) for p in g.logit_probabilities.tolist()],
             top_nodes=top_nodes(g, 25),
             tokens=[model.tokenizer.decode([i]) for i in g.input_tokens.tolist()])
    json.dump(s, open(f"/home/amit/.cache/vidyalai-scratch/p08/post/graph_{name}.json", "w"), indent=1)
    print({k: v for k, v in s.items() if k != "top_nodes"}, flush=True)
    for n in s["top_nodes"][:12]:
        print("  ", n)
    del g
    torch.cuda.empty_cache()
print(f"total {time.time()-t0:.0f}s")
