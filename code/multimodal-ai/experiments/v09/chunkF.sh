free -m | sed -n 2p
FB_BS=2 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python exp3_halluc.py turbo 2>&1 | grep -v -i "warn\|deprecat\|Loading weights\|attention mask\|generation_config"
