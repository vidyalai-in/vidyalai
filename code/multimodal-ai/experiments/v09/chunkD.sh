free -m | sed -n 2p
/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python exp1_wer.py turbo 2>&1 | grep -v -i "warn\|deprecat\|Loading weights\|attention mask\|generation_config"
