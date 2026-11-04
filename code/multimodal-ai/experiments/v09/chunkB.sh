free -m | sed -n 2p
for m in tiny base small turbo; do free -m | sed -n 2p;  /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python exp3_halluc.py $m 2>&1 | grep -v -i "warn\|deprecat\|Loading weights\|attention mask\|generation_config"; done
