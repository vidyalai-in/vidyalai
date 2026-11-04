import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

bs = G.BertScorer("FacebookAI/roberta-large", layer=17)
ref = "The Eiffel Tower is in Paris."
outs = ["The Eiffel Tower is in Paris.", "It stands in Paris, France.", "The Eiffel Tower is in Rome."]
print([round(f, 3) for f in bs.f1(outs, [[ref]] * 3)])
