import verify   # evals/lab/verify.py

pairs = [("\\dfrac{3}{4}", "\\frac 34"), ("x = 9", "9"), ("10080", "10,\\!080"),
         ("2√5", "2 \\sqrt{5}"), ("3 < \\lambda \\le 4", "(3,4]")]
for pred, gold in pairs:
    verdicts = {name: grade(pred, gold) for name, grade in verify.GRADERS.items()}
    print(f"{pred:18} {gold:12} {verdicts}")
