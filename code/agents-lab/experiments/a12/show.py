import json, sys
for l in open(sys.argv[1]):
    e=json.loads(l)
    if e['event']=='step':
        for b in e['content']:
            print(' ', b.get('type'), (b.get('text') or json.dumps(b.get('input')))[:160].replace(chr(10),' | '))
    if e['event']=='tool_results': print('  ->', e['results'][0]['content'][:100])
