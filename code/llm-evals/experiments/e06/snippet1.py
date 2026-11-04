import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

ref = "The Eiffel Tower is in Paris."
for out in ["The Eiffel Tower is in Paris.",
            "It stands in Paris, France.",
            "The Eiffel Tower is in Rome."]:
    print(f"{out:32s} EM={G.exact_match(out, ref):.0f}  F1={G.token_f1(out, ref):.2f}  "
          f"ROUGE-L={G.rouge(out, ref)['rougeL']:.2f}  BLEU={G.bleu(out, ref):5.1f}  chrF={G.chrf(out, ref):5.1f}")
