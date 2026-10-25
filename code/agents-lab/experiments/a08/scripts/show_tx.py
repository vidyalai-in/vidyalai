import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e['event'] == 'step':
        print(f"--- step {e['n']} out={e['output_tokens']} think={e.get('thinking_tokens')}")
        for b in e['content']:
            if b['type'] == 'text': print('TEXT', b['text'][:int(sys.argv[2]) if len(sys.argv)>2 else 400])
            elif b['type'] == 'tool_use': print('CALL', b['name'], json.dumps(b['input'])[:300])
            elif b['type'] == 'thinking': print('THINK', b.get('thinking','')[:500])
    elif e['event'] == 'tool_results':
        for r in e['results']: print('RESULT', ('ERR ' if r.get('is_error') else '') + str(r['content'])[:250].replace('\n',' | '))
