import json, sys
for l in open(sys.argv[1]):
    e=json.loads(l)
    if e['event']=='step':
        for b in e['content']:
            if b['type']=='text': print('  TEXT', b['text'][:600].replace(chr(10),' | '))
            elif b['type']=='tool_use': print('  CALL', b['name'], json.dumps(b['input']))
    if e['event']=='tool_results':
        for r in e['results']: print('   ->', r['content'][:110])
    if e['event'] in ('crash','resume'): print('==', e)
