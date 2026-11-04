# writes compact data strings for the toy figure generators
import json, numpy as np
p = json.load(open('/home/amit/.cache/vidyalai-scratch/v05/runs/toy_pics.json'))
fw = {k: np.round(np.array(v)[:500], 2).tolist() for k, v in p['forward'].items()}
tr = {m: np.round(np.array(p[f'traj_{m}'])[:, :24], 2).tolist() for m in ('ddpm', 'ddim')}
end = {m: np.round(np.array(p[f'end_{m}'])[:300], 2).tolist() for m in ('ddpm', 'ddim')}
json.dump({'fw': fw}, open('/home/amit/.cache/vidyalai-scratch/v05/figs/toy_fw.json', 'w'), separators=(',', ':'))
json.dump({'tr': tr, 'end': end}, open('/home/amit/.cache/vidyalai-scratch/v05/figs/toy_tr.json', 'w'), separators=(',', ':'))
