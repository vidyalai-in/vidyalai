import re,sys
t=open(sys.argv[1]).read(); n=int(sys.argv[3]) if len(sys.argv)>3 else 300
for m in list(re.finditer(sys.argv[2],t,flags=re.I))[:int(sys.argv[4]) if len(sys.argv)>4 else 6]:
    print('...'+t[max(0,m.start()-n):m.end()+n].replace('\n',' ')+'...\n')
