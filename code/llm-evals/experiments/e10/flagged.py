import json,sys
sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/evals')
from lab import safety_evals as S
from lab.grading import load_runs
X={r['id']:r for r in S.load_xstest()}
tag=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 200
H=load_runs(f'/home/amit/.cache/vidyalai-scratch/e10/runs/xstest_{tag}.jsonl'); J=load_runs('/home/amit/.cache/vidyalai-scratch/e10/runs/judges.jsonl')
for k,r in H.items():
    p=int(k.split('-')[1])
    if X[p]['label']!='safe': continue
    j=S.parse_judge(J[f'x3-{tag}-{k}']['text'])
    if j!='full_compliance': print(k,j[:4],'|',X[p]['prompt'][:55],'||',r['text'][:n].replace('\n',' '))
