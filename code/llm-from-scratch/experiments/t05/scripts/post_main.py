# Final checks on the 22M run's last checkpoint: loss on the whole validation shard, and two samples.
# usage (from fromscratch/): python post_main.py <ckpt.pt>
import sys, time
import torch
from data import ShardLoader
from model import GPT, SMALL
from tokenizer import load_production

model = GPT(SMALL).cuda()
model.load_state_dict(torch.load(sys.argv[1], map_location="cuda", weights_only=False)["model"])
model.eval()
val = ShardLoader("val", B=16, T=1024)
t0, total = time.time(), torch.zeros((), device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(val.n_batches):
        x, y = (t.cuda() for t in val.batch(i))
        total += model(x, y)
print(f"full val: {val.n_batches} micro-batches, {val.n_batches * 16 * 1024:,} tokens, "
      f"loss {(total / val.n_batches).item():.4f}, {time.time() - t0:.1f} s")
tok = load_production()
for prompt in ["The water cycle is", "In 1492, Christopher Columbus"]:
    ids = torch.tensor([tok.encode(prompt).ids], device="cuda")
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        greedy = model.generate(ids, 40, temperature=0)
        sampled = model.generate(ids, 40, temperature=0.8, top_k=50)
    print("GREEDY :", repr(tok.decode(greedy[0].tolist())))
    print("SAMPLED:", repr(tok.decode(sampled[0].tolist())))
