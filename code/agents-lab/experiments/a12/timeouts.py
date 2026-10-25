"""Bare and model_key arms: after each refund timeout the model saw, what did it do next for that order?"""
import json, sys
from collections import Counter
for arm in sys.argv[1:]:
    c = Counter(); dupsrc = Counter()
    for l in open(f'runs/refunds-{arm}.jsonl'):
        r = json.loads(l)
        if r['seed'] > 900: continue
        ev = [json.loads(x) for x in open(f'runs/tx/refunds-{arm}/{r["id"]}.jsonl')]
        calls = {}  # tool_use id -> (name, input)
        seq = []    # (name, order, result)
        for e in ev:
            if e['event'] == 'step':
                for b in e['content']:
                    if b['type'] == 'tool_use': calls[b['id']] = (b['name'], b['input'])
            if e['event'] == 'tool_results':
                for res in e['results']:
                    name, inp = calls[res['tool_use_id']]
                    seq.append((name, inp['order_id'], res['content'], inp.get('idempotency_key')))
        for i, (name, oid, out, key) in enumerate(seq):
            if name == 'refund' and out.startswith('ToolTimeout'):
                nxt = [(n, o, k) for n, o, _, k in seq[i+1:] if o == oid]
                first = nxt[0][0] if nxt else 'nothing'
                if first == 'refund' and arm == 'model_key':
                    first = 'refund, same key' if nxt[0][2] == key else 'refund, new key'
                c[first] += 1
        dupsrc['delivery'] += sum(1 for f in r['fault_log'] if f[2] == 'duplicate')
    print(arm, 'after a timeout, next action on that order:', dict(c))
