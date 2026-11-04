"""Download MMLU-Redux 2.0 (CC-BY-4.0) into one JSON file: evals/data/e02-redux.json."""
import json
from datasets import get_dataset_config_names, load_dataset

NAME = "edinburgh-dawg/mmlu-redux-2.0"
out = []
for cfg in sorted(get_dataset_config_names(NAME)):
    ds = load_dataset(NAME, cfg, split="test")
    for i, r in enumerate(ds):
        out.append(dict(id=f"{cfg}/{i}", subject=cfg, question=r["question"], choices=r["choices"],
                        answer=r["answer"], error_type=r["error_type"], source=r["source"],
                        correct_answer=r["correct_answer"], potential_reason=r["potential_reason"]))
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/evals/data/e02-redux.json", "w"))
print(len(out))
