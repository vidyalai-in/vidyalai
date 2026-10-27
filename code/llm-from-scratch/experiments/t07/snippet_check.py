from snapshots import load, complete

for step in (0, 25, 100, 250, 500, 1000, 2000, 3360):
    model = load(f"runs/main88/model_{step:05d}.pt")
    print(step, "greedy: ", complete(model, "The water cycle is", n_tokens=48))
    print(step, "sampled:", complete(model, "The water cycle is", n_tokens=48, temperature=0.8, top_k=50, seed=0))
