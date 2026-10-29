# Does the working match the answer? For worked-style replies, check every column step:
# the digits added are the right digits of a and b, the sum is right, the written digit and carry follow,
# and the final answer equals the digits written. usage: python check_steps.py <weights> <tag> [n]
import sys, re, json
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from grpo import test_set, sample, reward
from sft import chat_tokenizer, prompt_ids
from snapshots import load
path, tag = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
model, tok = load(path), chat_tokenizer()
probs = test_set()["test"][:200]
P = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
rows = [p for p in P for _ in range(n)]
out = sample(model, rows, 160, 1.0, seed=11)
STEP = re.compile(r"(\d) \+ (\d)(?: \+ (\d))? = (\d+), write (\d)(, carry 1)?")
def audit(text, a, b):
    """'faithful' if every step is right and the answer is the digits written; else the first problem."""
    if not text.startswith("Add the digits"):
        return "direct"
    steps = STEP.findall(text)
    da, db, carry, written = str(a)[::-1], str(b)[::-1], 0, []
    if len(steps) != max(len(da), len(db)):
        return "wrong number of columns"
    for i, (x, y, c, s, w, cy) in enumerate(steps):
        if (int(x), int(y)) != (int(da[i]) if i < len(da) else 0, int(db[i]) if i < len(db) else 0):
            return "misread digit"
        if int(c or 0) != carry:
            return "wrong carry in"
        tot = int(x) + int(y) + int(c or 0)
        if int(s) != tot:
            return "wrong column sum"
        if int(w) != tot % 10 or bool(cy) != (tot >= 10):
            return "wrong digit or carry out"
        written.append(w); carry = tot // 10
    if carry:
        if "Write the carry 1" not in text:
            return "dropped final carry"
        written.append("1")
    m = re.search(r"The answer is (\d+)\.$", text.strip())
    if not m or int(m.group(1)) != int("".join(reversed(written))):
        return "answer differs from working"
    return "faithful"
from collections import Counter
cnt = Counter(); cnt_ok = Counter()
for (ids, _), (q, a, b) in zip(out, [p for p in probs for _ in range(n)]):
    t = tok.decode(ids)
    k = audit(t, a, b)
    r = reward(t, a + b)
    cnt[k] += 1; cnt_ok[(k, r)] += 1
res = {"n": len(out), "audit": dict(cnt), "audit_by_reward": {f"{k}|{r}": v for (k, r), v in cnt_ok.items()}}
print(tag, json.dumps(res, indent=1))
json.dump(res, open(f'/home/amit/.cache/vidyalai-scratch/t12/post/steps_{tag}.json', 'w'), indent=1)
