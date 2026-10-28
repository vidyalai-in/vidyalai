import sys; sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
from rm import RewardModel
rm = RewardModel()
print(rm.tok.decode(rm.encode("What is 2+2?", "4")))
print(rm.score([("What is the capital of France?", "The capital of France is Paris."),
                ("What is the capital of France?", "The capital of France is the capital of France. Its capital is the capital of France."),
                ("What is the capital of France?", "Paris is the capital of Germany.")]))
