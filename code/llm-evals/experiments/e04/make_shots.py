"""Few-shot pools for Part 4: MMLU dev (the standard 5 per subject) and two draws of 5 from validation."""
import json, random
import pandas as pd
P = '/home/amit/.cache/vidyalai-scratch/hf/hub/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe/all/'
dev = pd.read_parquet(P + 'dev-00000-of-00001.parquet')
val = pd.read_parquet(P + 'validation-00000-of-00001.parquet')
def rows(df):
    return [dict(question=r.question, choices=list(r.choices), answer=int(r.answer)) for r in df.itertuples()]
out = {}
for subj in sorted(dev.subject.unique()):
    v = rows(val[val.subject == subj])
    out[subj] = {"dev": rows(dev[dev.subject == subj]),
                 "s1": random.Random(1000 + len(subj)).sample(v, 5),
                 "s2": random.Random(2000 + len(subj)).sample(v, 5)}
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/evals/data/e04-shots.json', 'w'))
print(len(out), sum(len(x['dev']) for x in out.values()))
