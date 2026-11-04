"""Long-context eval helpers for the Measuring LLMs series (Part 9, "Long-context evals").

A haystack is filler text (six Jane Austen novels from Project Gutenberg, public domain in the US;
the text is read from the scratch folder and never committed). Into each haystack we insert:

* K "character" needles in NoLiMa's style (Modarressi et al. 2025, arXiv:2502.05167):
  "Actually, Yuki lives next to the Semper Opera House." Each needle is asked about twice:
    - literal:     "Which character lives next to the Semper Opera House?"  (shares the landmark words)
    - non-literal: "Which character has been to Dresden?"  (no word in common; needs one hop of
                   world knowledge, landmark -> city)
* C variable-tracking chains in RULER's style (Hsieh et al. 2024, arXiv:2404.06654):
  "VAR QXKZP = 48213." then "VAR BMWTA = VAR QXKZP." and so on, spread through the text. The question
  asks for every variable that ends up holding one value.

Every request asks one question about one haystack, so a haystack is a cacheable prefix and the
question is the only part that changes. Nothing here calls an API.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

CORPUS_DIR = Path("/home/amit/.cache/vidyalai-scratch/e09/data/gutenberg")
BOOKS = ["pg1342", "pg161", "pg158", "pg141", "pg105", "pg121"]  # P&P, S&S, Emma, Mansfield, Persuasion, Northanger

# (landmark, city). Checked: no landmark, city or name below occurs in the six novels.
LANDMARKS = [
    ("Kiasma museum", "Helsinki"), ("Kiyomizu-dera temple", "Kyoto"), ("Atomium", "Brussels"),
    ("Petronas Towers", "Kuala Lumpur"), ("Christ the Redeemer statue", "Rio de Janeiro"),
    ("Sagrada Familia", "Barcelona"), ("CN Tower", "Toronto"), ("Burj Khalifa", "Dubai"),
    ("Hagia Sophia", "Istanbul"), ("Charles Bridge", "Prague"), ("Little Mermaid statue", "Copenhagen"),
    ("Space Needle", "Seattle"), ("Table Mountain cableway", "Cape Town"), ("Taj Mahal", "Agra"),
    ("Brandenburg Gate", "Berlin"), ("Colosseum", "Rome"), ("Acropolis", "Athens"),
    ("Shwedagon Pagoda", "Yangon"), ("Marina Bay Sands", "Singapore"), ("Hallgrimskirkja church", "Reykjavik"),
]
NAMES = ["Yuki", "Mateo", "Priya", "Kofi", "Ingrid", "Tariq", "Mei", "Diego", "Aiko", "Ravi",
         "Zara", "Leilani", "Omar", "Sven", "Amara", "Kenji", "Farid", "Noor", "Anika", "Bao"]

# NoLiMa's prompt template without chain of thought (arXiv:2502.05167, Appendix C, Table 9).
PREAMBLE = "You will answer a question based on the following book snippet:\n\n"
POSTAMBLE = ("\n\nUse the information provided in the book snippet to answer the question. Your answer should be "
             "short and based on either explicitly stated facts or strong, logical inferences.")
_ONLY = "\nReturn only the final answer with no additional explanation or reasoning."
Q_CHAR_LITERAL = "Question: Which character lives next to the {landmark}?" + _ONLY
Q_CHAR_ONEHOP = "Question: Which character has been to {city}?" + _ONLY
Q_VT = ("Question: Find all variables that are assigned the value {value} in the book snippet, directly or through "
        "a chain of assignments (VAR B = VAR A, where VAR A = {value}). List every such variable, separated by commas."
        + _ONLY)
Q_KNOW = "In which city is the {landmark}? Answer with only the city name."


# ---------------------------------------------------------------- corpus ---------------------------

def load_corpus():
    """The six novels, Gutenberg header and footer removed, as one list of paragraphs."""
    paras = []
    for b in BOOKS:
        t = (CORPUS_DIR / f"{b}.txt").read_text(encoding="utf-8-sig")
        t = t.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1].split("***", 1)[1]
        t = t.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
        for p in re.split(r"\n\s*\n", t):
            p = " ".join(p.split())
            if p:
                paras.append(p)
    return paras


def filler(paras, n_chars, start_para=0):
    """About n_chars of consecutive paragraphs starting at start_para (wrapping around)."""
    out, size, i = [], 0, start_para
    while size < n_chars:
        p = paras[i % len(paras)]
        out.append(p)
        size += len(p) + 2
        i += 1
    text = "\n\n".join(out)
    return text[:n_chars].rsplit(" ", 1)[0]


# ---------------------------------------------------------------- needles --------------------------

def _var(rng):
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))


def make_needles(rng, k, c, hops=4):
    """K character needles and C variable chains (each: one value, then `hops` copies)."""
    lm = rng.sample(LANDMARKS, k)
    names = rng.sample(NAMES, k)
    chars = [dict(name=n, landmark=l, city=city) for n, (l, city) in zip(names, lm)]
    chains, used = [], set()
    for _ in range(c):
        value = str(rng.randint(10000, 99999))
        vs = []
        while len(vs) < hops + 1:
            v = _var(rng)
            if v not in used:
                used.add(v)
                vs.append(v)
        stmts = [f"VAR {vs[0]} = {value}."] + [f"VAR {vs[j]} = VAR {vs[j - 1]}." for j in range(1, hops + 1)]
        chains.append(dict(value=value, vars=vs, statements=stmts))
    return chars, chains


def _boundary(text, pos):
    """The nearest sentence end at or before pos (a '. ', '? ', '! ' or paragraph break), else 0."""
    if pos >= len(text):
        return len(text)
    m = None
    for m in re.finditer(r"[.!?][\"']?\s", text[max(0, pos - 2000):pos]):
        pass
    return max(0, pos - 2000) + m.end() if m else 0


def insert(text, items):
    """Insert sentences at fractional depths. items: list of (depth in [0, 1], sentence, key).
    Returns the new text and {key: actual depth as a fraction of the final text}."""
    spots = sorted(((_boundary(text, int(d * len(text))), i) for i, (d, _, _) in enumerate(items)), reverse=True)
    for pos, i in spots:  # right to left, so earlier positions stay valid
        s = items[i][1]
        text = text[:pos] + s + " " + text[pos:] if pos < len(text) else text + " " + s
    # actual positions in the final text
    depth = {}
    for d, s, key in items:
        depth[key] = text.find(s) / max(1, len(text))
    return text, depth


def build_haystack(paras, n_chars, start_para, seed, k=6, c=3, char_depths=None):
    """One haystack: filler of n_chars with K character needles at char_depths and C chains whose
    statements sit at random depths (in chain order). Returns a dict with the text and the questions."""
    rng = random.Random(seed)
    chars, chains = make_needles(rng, k, c)
    if char_depths is None:
        char_depths = list(np.linspace(0, 1, k))
    items = []
    for ch, d in zip(chars, char_depths):
        ch["needle"] = f"Actually, {ch['name']} lives next to the {ch['landmark']}."
        items.append((float(d), ch["needle"], ("char", ch["name"])))
    for j, chn in enumerate(chains):
        ds = sorted(rng.uniform(0.02, 0.98) for _ in chn["statements"])
        chn["depths_planned"] = ds
        for h, (d, s) in enumerate(zip(ds, chn["statements"])):
            items.append((d, s, ("vt", j, h)))
    text = filler(paras, n_chars, start_para) if n_chars > 0 else ""
    if n_chars == 0:  # control: the needles alone, in depth order
        text = " ".join(s for _, s, _ in sorted(items, key=lambda x: x[0]))
        depth = {key: d for d, _, key in items}
    else:
        text, depth = insert(text, items)
    for ch in chars:
        ch["depth"] = depth[("char", ch["name"])]
    for j, chn in enumerate(chains):
        chn["depths"] = [depth[("vt", j, h)] for h in range(len(chn["statements"]))]
    return dict(text=text, chars=chars, chains=chains, seed=seed, start_para=start_para)


def questions(hay):
    """Every question for one haystack: (qid, task, question text, gold, depth)."""
    qs = []
    for ch in hay["chars"]:
        qs.append((f"lit-{ch['name']}", "literal", Q_CHAR_LITERAL.format(**ch), ch["name"], ch["depth"]))
        qs.append((f"hop-{ch['name']}", "onehop", Q_CHAR_ONEHOP.format(**ch), ch["name"], ch["depth"]))
    for j, chn in enumerate(hay["chains"]):
        qs.append((f"vt-{j}", "vt", Q_VT.format(value=chn["value"]), chn["vars"], float(np.mean(chn["depths"]))))
    return qs


# ---------------------------------------------------------------- prompts --------------------------

def claude_params(hay_text, question, model="claude-haiku-5-5", max_tokens=48, ttl="1h", cache=True):
    """A Messages request: the haystack as a cached first block, the question as a second block.
    Thinking is off (Haiku 5.5 thinks by default when `thinking` is omitted)."""
    block = {"type": "text", "text": PREAMBLE + hay_text + POSTAMBLE}
    if cache:
        block["cache_control"] = {"type": "ephemeral", "ttl": ttl} if ttl == "1h" else {"type": "ephemeral"}
    return dict(model=model, max_tokens=max_tokens, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": [block, {"type": "text", "text": question}]}])


def hf_prompt_parts(tok, hay_text, question):
    """(prefix ids, suffix ids) for a chat model, split so the haystack prefix can be prefilled once
    and reused for every question (local prefix caching)."""
    full = tok.apply_chat_template([{"role": "user", "content": PREAMBLE + hay_text + POSTAMBLE + "\n\n" + question}],
                                   tokenize=False, add_generation_prompt=True)
    cut = full.index(POSTAMBLE) + len(POSTAMBLE)
    pre = tok(full[:cut], add_special_tokens=False).input_ids
    suf = tok(full[cut:], add_special_tokens=False).input_ids
    return pre, suf


# ---------------------------------------------------------------- grading --------------------------

def grade_name(reply, gold, candidates=NAMES):
    """Correct if the gold name appears as a word and no other inserted name does."""
    found = {n for n in candidates if re.search(rf"\b{re.escape(n)}\b", reply or "")}
    return int(gold in found and len(found) == 1)


def final_line(reply):
    lines = [l for l in (reply or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def grade_vt(reply, gold_vars):
    """(all correct, recall), read from the reply's last non-empty line (a reply that reasons first
    ends with its list). All correct = that line names exactly the chain's variables."""
    found = set(re.findall(r"\b[A-Z]{5}\b", final_line(reply)))
    gold = set(gold_vars)
    return int(found == gold), len(found & gold) / len(gold)


HEDGES = re.compile(r"not (state|say|mention|explicit)|doesn't|does not|can't|cannot|no character|none of|no one|unclear|not sure", re.I)


def hedged(reply):
    """True if the reply hedges or denies (e.g. names the character but says the text doesn't say so)."""
    return bool(HEDGES.search(reply or ""))


def grade_city(reply, city):
    return int(city.lower().split()[0] in (reply or "").lower())
