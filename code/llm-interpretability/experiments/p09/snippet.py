import sys, torch
sys.path.insert(0, "interp")                       # the series' shared lab/ folder
from lab.steering import Steerer

s = Steerer("interp/models/gemma-2-2b-it-bf16")
r = torch.load("p09/runs/gemma_dirs.pt")["r"][14, 3]   # layer 14, at the "model" token
prompt = s.chat(["Give me a recipe for banana pancakes."])
for name, edit in [("unedited", None), ("refusal added", lambda m: m.add(14, r))]:
    with s.edits(edit):
        print(name, "|", s.generate(prompt, max_new_tokens=30)[0])
