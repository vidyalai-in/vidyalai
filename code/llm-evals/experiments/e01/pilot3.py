import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
cl = ledger.client()
m = V.load(f"{D}/mmlu-1140.json")
V.SYSTEM = "You are taking a multiple-choice test. Your entire reply must be one letter, the letter of your answer, with no explanation."
V.SYSTEM_CO = ("You are taking a multiple-choice test, but the question has been hidden: you only see the answer options. "
               "Guess which option is most likely the correct answer. Your entire reply must be one letter, the letter of your guess, with no explanation.")
for i, kind, model in [(12, "full", "claude-sonnet-5-5"), (40, "choices_only", "claude-sonnet-5-5"), (300, "full", "claude-sonnet-5-5"), (40, "choices_only", "claude-haiku-5-5")]:
    p = V.request(m[i], kind, model)
    r = ledger.create(cl, "e01", note="e01 pilot3", **p)
    print(model, kind, r.stop_reason, repr("".join(c.text for c in r.content if c.type=="text")), m[i]["answer"])
