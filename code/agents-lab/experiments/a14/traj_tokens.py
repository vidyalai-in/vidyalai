"""Part 14 measurement (no API calls): what a coding-agent trajectory looks like as training data.

Takes the Claude Haiku 5.5 transcripts from Part 8 (full tool kit, 28 tasks x 3 repeats), renders each
one with the Qwen3 chat template (an open model's format, so it can be tokenized locally), and counts
which tokens a loss mask would train on: tokens the model wrote (assistant turns) versus tokens the
environment wrote (tool results) and the prompt (system, tools, issue).

Thinking blocks are left out: Haiku's thinking arrives summarized/encrypted, so its text isn't available.
"""
import ast
import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from transformers import AutoTokenizer  # noqa: E402

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08/runs")
OUT = Path("/home/amit/.cache/vidyalai-scratch/a14/traj_tokens.json")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def lit(s):
    return ast.literal_eval(s) if isinstance(s, str) else s


def to_messages(path):
    ev = [json.loads(l) for l in path.read_text().splitlines()]
    head = ev[0]
    system = "".join(b["text"] for b in lit(head["system"]))
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                "parameters": t["input_schema"]}} for t in lit(head["tools"])]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": head["task"]}]
    for e in ev[1:]:
        if e["event"] == "step":
            blocks = lit(e["content"])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"type": "function", "function": {"name": b["name"], "arguments": b["input"]}}
                     for b in blocks if b.get("type") == "tool_use"]
            m = {"role": "assistant", "content": text}
            if calls:
                m["tool_calls"] = calls
            msgs.append(m)
        elif e["event"] == "tool_results":
            for r in lit(e["results"]):
                c = r["content"]
                if isinstance(c, list):
                    c = "".join(b.get("text", "") for b in c)
                msgs.append({"role": "tool", "content": c})
    return msgs, tools


def count(msgs, tools):
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    # Character spans of assistant turns: from after "<|im_start|>assistant\n" to the end of "<|im_end|>".
    spans, start_tag, end_tag, i = [], "<|im_start|>assistant\n", "<|im_end|>", 0
    while True:
        a = text.find(start_tag, i)
        if a < 0:
            break
        b = text.find(end_tag, a) + len(end_tag)
        spans.append((a + len(start_tag), b))
        i = b
    # Tool results sit inside user turns as <tool_response>...</tool_response>.
    tspans, i = [], 0
    while True:
        a = text.find("<tool_response>", i)
        if a < 0:
            break
        b = text.find("</tool_response>", a) + len("</tool_response>")
        tspans.append((a, b))
        i = b
    model = env = 0
    for (s, e) in enc["offset_mapping"]:
        if any(a <= s < b for a, b in spans):
            model += 1
        elif any(a <= s < b for a, b in tspans):
            env += 1
    total = len(enc["input_ids"])
    first_user_end = text.find("<|im_end|>", text.find("<|im_start|>user")) + len("<|im_end|>")
    prompt = len(tok(text[:first_user_end], add_special_tokens=False)["input_ids"])
    return {"total": total, "model": model, "env": env, "prompt": prompt, "other": total - model - env - prompt}


rows = []
for p in sorted((A08 / "transcripts").glob("*-full-haiku-medium-r*.jsonl")):
    msgs, tools = to_messages(p)
    r = count(msgs, tools)
    r["file"] = p.name
    r["turns"] = sum(m["role"] == "assistant" for m in msgs)
    rows.append(r)

OUT.write_text(json.dumps(rows, indent=1))
n = len(rows)
tot = {k: sum(r[k] for r in rows) for k in ("total", "model", "env", "prompt", "other")}
print(f"{n} trajectories (Haiku 5.5, full kit), Qwen3 tokenizer")
for k in ("prompt", "model", "env", "other"):
    print(f"  {k:7s} {tot[k]:8d} tokens  {100 * tot[k] / tot['total']:5.1f}%")
share = [r["model"] / r["total"] for r in rows]
print(f"  per-trajectory share trained on: median {100 * st.median(share):.1f}%, "
      f"min {100 * min(share):.1f}%, max {100 * max(share):.1f}%")
print(f"  median tokens per trajectory {st.median(r['total'] for r in rows):.0f}, "
      f"median assistant turns {st.median(r['turns'] for r in rows):.0f}")
ex = next(r for r in rows if r["file"].startswith("cachetools-rr-empty-full-haiku-medium-r0"))
print("  example", ex)
sys.exit(0)
