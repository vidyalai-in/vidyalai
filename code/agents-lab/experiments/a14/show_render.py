import sys, os
sys.argv=['x']
os.environ["CUDA_VISIBLE_DEVICES"]=""
exec(open('/home/amit/.cache/vidyalai-scratch/a14/traj_tokens.py').read().split("rows = []")[0])
msgs, tools = to_messages(A08 / "transcripts/cachetools-rr-empty-full-haiku-medium-r0.jsonl")
t = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
i = t.find("<|im_start|>assistant")
print(t[i:i+1800])
