"""IOI prompt set for Part 5, built from the 15 templates of Wang et al. (2022), Appendix E
(BABA order as listed, plus the ABBA variant of each), with random single-token names,
places and objects. Corruptions: STR-S2 (S2 replaced by IO, Zhang & Nanda 2023), ABC
(three unrelated names in the IO, S1, S2 slots, Wang et al.'s p_ABC), and the clean prompt
itself for Gaussian noise on S2's embedding (applied in the experiment)."""
import random

TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
]
NAMES = """Michael Christopher Jessica Matthew Ashley Jennifer Joshua Amanda Daniel David James Robert
John Joseph Andrew Ryan Brandon Jason Justin Sarah William Jonathan Stephanie Brian Nicole Nicholas
Anthony Heather Eric Elizabeth Adam Megan Melissa Kevin Steven Thomas Timothy Christina Kyle Rachel
Laura Lauren Amber Brittany Danielle Richard Kimberly Jeffrey Amy Crystal Michelle Tiffany Jeremy
Benjamin Mark Emily Aaron Charles Rebecca Jacob Stephen Patrick Sean Erin Zachary Jamie Kelly Samantha
Nathan Sara Dustin Paul Angela Tyler Scott Katherine Andrea Gregory Erica Mary Travis Lisa Kenneth
Bryan Lindsey Kristen Jose Alexander Jesse Katie Lindsay Shannon Vanessa Courtney Christine Alicia
Cody Allison Bradley Samuel George Peter Simon Victoria Grace Anna Alice Henry Frank Louis Jack""".split()
PLACES = "store garden school house restaurant hospital office station park church library market beach river cafe museum hotel theater bank airport".split()
OBJECTS = "ring kiss bone basketball computer necklace drink snack phone bottle cake book bag pen ball box letter key cup knife".split()


def single_token(tok, words):
    return [w for w in words if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]


def build(tok, n=500, seed=0):
    """Returns a list of dicts with clean/str/abc texts, the IO and S names, and the template id."""
    names, places, objects = single_token(tok, NAMES), single_token(tok, PLACES), single_token(tok, OBJECTS)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i % (2 * len(TEMPLATES))
        tpl = TEMPLATES[t // 2]
        if t % 2 == 1:   # ABBA: swap the first [B] and [A]
            tpl = tpl.replace("[B] and [A]", "[A] and [B]", 1)
        io, s, c1, c2, c3 = rng.sample(names, 5)
        place, obj = rng.choice(places), rng.choice(objects)
        fill = lambda a, b: tpl.replace("[PLACE]", place).replace("[OBJECT]", obj).replace("[A]", a).replace("[B]", b)
        clean = fill(io, s)
        # STR-S2: the second mention of S becomes IO, so the right answer flips to S
        k = clean.rfind(s)
        str_s2 = clean[:k] + io + clean[k + len(s):]
        # ABC: IO, S1, S2 replaced by three unrelated names (no name is repeated)
        abc_tpl = tpl.replace("[B]", "[C]", 1) if t % 2 == 0 else tpl
        if t % 2 == 1:
            abc_tpl = tpl.replace("[B]", "[C]", 1)
        abc = abc_tpl.replace("[PLACE]", place).replace("[OBJECT]", obj).replace("[A]", c1).replace("[C]", c2).replace("[B]", c3)
        out.append(dict(clean=clean, str_s2=str_s2, abc=abc, io=io, s=s, template=t // 2, order="BABA" if t % 2 == 0 else "ABBA"))
    return out, dict(n_names=len(names), n_places=len(places), n_objects=len(objects))


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    data, info = build(tok, 60)
    print(info)
    for d in data[:4] + data[28:32]:
        print(d["order"], "|", d["clean"], "|", d["str_s2"], "|", d["abc"])
