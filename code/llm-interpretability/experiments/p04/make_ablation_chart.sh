cd /home/amit/AlpsCode/ai/vidyalai
D=$(/home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python /home/amit/.cache/vidyalai-scratch/p04/scripts/build_ablation_data.py)
python3 - "$D" <<'PY'
import sys
t=open('/home/amit/.cache/vidyalai-scratch/p04/scripts/ablation_chart_template.py').read().replace('__DATA__',sys.argv[1].replace('null','None'))
open('diagrams/p04-ablation.py','w').write(t)
PY
python3 diagrams/p04-ablation.py public/diagrams/p04-ablation
node -e "require('sharp')('public/diagrams/p04-ablation.svg',{density:144}).png().toFile('public/diagrams/p04-ablation.png')"
node -e "require('sharp')('public/diagrams/p04-ablation-dark.svg',{density:144}).png().toFile('/home/amit/.cache/vidyalai-scratch/p04/post/p04-ablation-dark.png')"
