import re,sys
s=open(sys.argv[1]).read().replace('\n',' ')
sents=re.split(r'(?<=[.!?])\s+',s)
pat=re.compile(sys.argv[2],re.I)
n=0
for i,x in enumerate(sents):
    if pat.search(x):
        print(f"[{i}] {x[:500]}"); n+=1
        if n>=int(sys.argv[3] if len(sys.argv)>3 else 40): break
