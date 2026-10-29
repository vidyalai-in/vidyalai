"""Write the self-contained chart generators diagrams/p02-*.py from p02/post/summary.json.
Each generator = the line-chart template (scripts/linechart.py) + a DATA literal + a spec."""
import json, os
REPO = "/home/amit/AlpsCode/ai/vidyalai"
S = json.load(open("/home/amit/.cache/vidyalai-scratch/p02/post/summary.json"))
TPL = open("/home/amit/.cache/vidyalai-scratch/p02/scripts/linechart.py").read()
TPL = TPL.split("\n", 2)[2]  # drop the template's own header comment


def r(x, k=3): return round(x, k)


def emit(name, header, data, spec_code):
    src = (f"# Generates public/diagrams/{name}.svg and {name}-dark.svg. {header}\n"
           f"# Usage: python3 diagrams/{name}.py public/diagrams/{name}, then render the PNG with sharp at density 144.\n"
           "# Palette and approach copied from diagrams/g04-roofline.py. Data: mean and 95% bootstrap CI over held-out documents.\n"
           + TPL + "\n\nDATA = " + json.dumps(data) + "\n\n" + spec_code + "\nwrite(SPEC, sys.argv[1])\n")
    open(f"{REPO}/diagrams/{name}.py", "w").write(src)
    print("wrote", name)


# 1. GPT-2 small: KL to the final distribution by layer, logit lens vs bias-only vs tuned lens
g = S["gpt2"]
data = {n: {"m": [r(t[0]) for t in g[n]["kl_bits"][:12]], "lo": [r(t[1]) for t in g[n]["kl_bits"][:12]],
            "hi": [r(t[2]) for t in g[n]["kl_bits"][:12]]} for n in ["logit", "bias", "muon"]}
spec = '''SPEC = {
  'xlim': (-0.4, 11.4), 'ylim': (0.1, 100), 'logy': True,
  'yticks': [0.1, 0.3, 1, 3, 10, 30, 100],
  'xticks': [(0, 'emb')] + [(i, str(i)) for i in range(1, 12)],
  'xlabel': 'Read after block (emb = embeddings only)',
  'ylabel': 'KL to final output (bits, log scale)',
  'series': [
    {'style': 'ink', 'label': 'Logit lens', 'pts': list(enumerate(DATA['logit']['m'])), 'lo': DATA['logit']['lo'], 'hi': DATA['logit']['hi'], 'label_at': (4.2, 13.5)},
    {'style': 'mut', 'label': 'Bias only', 'pts': list(enumerate(DATA['bias']['m'])), 'lo': DATA['bias']['lo'], 'hi': DATA['bias']['hi'], 'label_at': (2.3, 4.1)},
    {'style': 'acc', 'label': 'Tuned lens', 'pts': list(enumerate(DATA['muon']['m'])), 'lo': DATA['muon']['lo'], 'hi': DATA['muon']['hi'], 'label_at': (1.6, 1.6)},
  ]}
'''
emit("p02-lens-kl", "GPT-2 small, 304 held-out FineWeb-Edu documents x 510 positions.", data, spec)

# 2. Residual-stream norm by depth for the three models (log scale), position 0 excluded
models = [k for k in ["gpt2", "pythia-160m", "qwen2.5-0.5b"] if k in S]
data = {k: [r(t[0], 2) for t in S[k]["norm"]] for k in models}
if len(models) == 3:
    spec = '''SPEC = {
  'xlim': (-0.02, 1.02), 'ylim': (0.3, 1000), 'logy': True,
  'yticks': [0.3, 1, 3, 10, 30, 100, 300, 1000],
  'xticks': [(0, 'embeddings'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1, 'last block')],
  'xlabel': 'Depth (fraction of the blocks run)',
  'ylabel': 'Mean residual-stream norm (log scale)',
  'series': [
    {'style': 'acc', 'label': 'GPT-2 small', 'pts': [(i / (len(DATA['gpt2']) - 1), v) for i, v in enumerate(DATA['gpt2'])], 'label_at': (0.08, 720), 'swatch': True},
    {'style': 'ink', 'label': 'Pythia-160M', 'pts': [(i / (len(DATA['pythia-160m']) - 1), v) for i, v in enumerate(DATA['pythia-160m'])], 'label_at': PY_LABEL, 'swatch': True},
    {'style': 'mut', 'label': 'Qwen2.5-0.5B', 'pts': [(i / (len(DATA['qwen2.5-0.5b']) - 1), v) for i, v in enumerate(DATA['qwen2.5-0.5b'])], 'label_at': QW_LABEL, 'swatch': True},
  ]}
'''
    emit("p02-norm-growth", "Three models, the same 304 held-out documents, first position excluded.", data,
         "PY_LABEL = (0.08, 480)\nQW_LABEL = (0.08, 322)\n" + spec)

# 3. Top-1 agreement with the final output by depth: logit lens on three models, tuned lens on Pythia
if len(models) == 3:
    data = {"gpt2": [r(t[0], 2) for t in S["gpt2"]["logit"]["top1_pct"]],
            "pythia": [r(t[0], 2) for t in S["pythia-160m"]["logit"]["top1_pct"]],
            "qwen": [r(t[0], 2) for t in S["qwen2.5-0.5b"]["logit"]["top1_pct"]],
            "pythia_tuned": [r(t[0], 2) for t in S["pythia-160m"]["muon"]["top1_pct"]]}
    spec = '''def pts(k): v = DATA[k]; return [(i / (len(v) - 1), x) for i, x in enumerate(v)]
SPEC = {
  'xlim': (-0.02, 1.02), 'ylim': (0, 100),
  'yticks': [0, 20, 40, 60, 80, 100], 'yfmt': '{:g}%',
  'xticks': [(0, 'embeddings'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1, 'last block')],
  'xlabel': 'Depth (fraction of the blocks run)',
  'ylabel': 'Top guess matches final output',
  'series': [
    {'style': 'mut', 'label': 'Qwen2.5-0.5B, logit lens', 'pts': pts('qwen'), 'label_at': QW, 'swatch': True},
    {'style': 'mutsolid', 'label': 'Pythia-160M, logit lens', 'pts': pts('pythia'), 'label_at': PY, 'swatch': True},
    {'style': 'ink', 'label': 'GPT-2 small, logit lens', 'pts': pts('gpt2'), 'label_at': G2, 'swatch': True},
    {'style': 'acc', 'label': 'Pythia-160M, tuned lens', 'pts': pts('pythia_tuned'), 'label_at': PT, 'swatch': True},
  ]}
'''
    emit("p02-model-compare", "Logit lens on three models, tuned lens on Pythia-160M; same 304 held-out documents.", data,
         "PT = (0.08, 93)\nG2 = (0.08, 86)\nPY = (0.08, 79)\nQW = (0.08, 72)\n" + spec)
