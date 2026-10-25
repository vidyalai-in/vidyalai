import re,sys
s=open(sys.argv[1]).read().split('---',2)[2]
s=re.sub(r'```.*?```','',s,flags=re.S)
s='\n'.join(l for l in s.splitlines() if not l.startswith('|') and not l.startswith('!['))
s=s.split('## Further reading')[0]
secs=re.split(r'\n(## .*)\n',s)
print('total',len(s.split()))
for i in range(1,len(secs),2): print(len(secs[i+1].split()), secs[i])
