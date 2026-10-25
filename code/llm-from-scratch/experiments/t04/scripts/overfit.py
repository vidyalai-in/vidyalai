# Drive one batch of real text to near-zero loss: proves gradients reach every parameter.
import json, time, torch
from common import heldout_batch
from model import GPT, MAIN
from tokenizer import load_production

torch.manual_seed(1337)
model = GPT(MAIN).cuda()
x, y = heldout_batch(B=4, T=1024)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
t0 = time.time()
log = open("/home/amit/.cache/vidyalai-scratch/t04/logs/overfit_steps.jsonl", "w")
for step in range(301):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    if step == 0:
        no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        print("parameters without gradient:", no_grad)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    log.write(f'{{"step": {step}, "loss": {loss.item():.5f}, "grad_norm": {norm.item():.4f}}}\n')
    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  grad norm {norm.item():.3f}  {time.time() - t0:.1f}s", flush=True)
    if loss.item() < 0.005:
        print(f"step {step:3d}  loss {loss.item():.4f}  stop"); break
torch.save(model.state_dict(), "/home/amit/.cache/vidyalai-scratch/t04/overfit.pt")
model.eval()
tok = load_production()
prompt = x[:1, :16]
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.generate(prompt, max_new_tokens=48, temperature=0)
print("PROMPT  ", repr(tok.decode(prompt[0].tolist())))
print("MODEL   ", repr(tok.decode(out[0, 16:].tolist())))
print("ACTUAL  ", repr(tok.decode(x[0, 16:64].tolist())))
print("exact match:", torch.equal(out[0, 16:], x[0, 16:64]))
