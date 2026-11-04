"""Request builders for Part 4's Claude experiments (Haiku 5.5, thinking disabled explicitly (omitting it lets Haiku 5.5 think), effort not set)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq

ITEMS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
SHOTS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e04-shots.json"))
SYS = "Reply with only the label of the correct option."

# condition -> (format, rotation, shots, system prompt?, max_tokens)
CONDS = {
    "raw":   ("mmlu", 0, 0, False, 768),
    "raw5":  ("mmlu", 0, 5, False, 768),
    "mmlu":  ("mmlu", 0, 0, True, 64),
    "rep":   ("mmlu", 0, 0, True, 64),
    "paren": ("paren", 0, 0, True, 64),
    "colon": ("colon", 0, 0, True, 64),
    "num":   ("num", 0, 0, True, 64),
    "lower": ("lower", 0, 0, True, 64),
    "rot1":  ("mmlu", 1, 0, True, 64),
    "rot2":  ("mmlu", 2, 0, True, 64),
    "rot3":  ("mmlu", 3, 0, True, 64),
    "mmlu5": ("mmlu", 0, 5, True, 64),
}


def params(cond, item, model="claude-haiku-5-5"):
    fname, r, k, sys_on, mt = CONDS[cond]
    it = mcq.rotate(item, r)
    shared, own = mcq.prompt(mcq.FORMATS[fname], it, SHOTS[item["subject"]]["dev"][:k])
    # Thinking off: Haiku 5.5 takes {"type": "disabled"}; Sonnet 5.5 rejects that and asks for "between_tools"
    off = {"type": "between_tools"} if model == "claude-sonnet-5-5" else {"type": "disabled"}
    p = dict(model=model, max_tokens=mt, thinking=off,
             messages=[{"role": "user", "content": shared + own}])
    if sys_on:
        p["system"] = SYS
    return p, it
