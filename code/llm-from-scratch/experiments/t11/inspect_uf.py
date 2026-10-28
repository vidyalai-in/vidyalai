# Inspect UltraFeedback binarized: ties, turns, lengths in our tokenizer, code share.
import sys, json, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import chat_tokenizer, render
D = 'data/ultrafeedback_binarized'
tok = chat_tokenizer()
for split in ('test_prefs', 'train_prefs'):
    rows = pq.read_table(f'{D}/{split}.parquet').to_pylist()
    n = len(rows)
    ties = sum(r['score_chosen'] == r['score_rejected'] for r in rows)
    turns = [len(r['chosen']) for r in rows]
    same = sum(r['chosen'][-1]['content'].strip() == r['rejected'][-1]['content'].strip() for r in rows)
    code = sum(any('{' in m['content'] for m in r['chosen'] + r['rejected']) for r in rows)
    sub = rows[:3000]
    pl = [len(tok.encode(r['prompt']).ids) for r in sub]
    cl = [len(tok.encode(r['chosen'][-1]['content']).ids) for r in sub]
    rl = [len(tok.encode(r['rejected'][-1]['content']).ids) for r in sub]
    sc = [r['score_chosen'] for r in rows]; sr = [r['score_rejected'] for r in rows]
    print(split, n, 'ties', ties, 'turns', dict(zip(*np.unique(turns, return_counts=True))), 'identical', same, 'has{', code)
    print(' prompt med', np.median(pl), 'chosen med', np.median(cl), 'mean', np.mean(cl), 'rejected med', np.median(rl), 'mean', np.mean(rl))
    print(' chosen longer frac', np.mean(np.array(cl) > np.array(rl)), 'score chosen mean', np.mean(sc), 'rej', np.mean(sr))
    print(' empty', sum(not r['chosen'][-1]['content'].strip() or not r['rejected'][-1]['content'].strip() for r in rows))
