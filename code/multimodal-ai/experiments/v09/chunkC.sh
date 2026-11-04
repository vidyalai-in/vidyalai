free -m | sed -n 2p
/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python exp2_noise.py turbo 2>&1 | grep -v -i "warn\|deprecat\|Loading weights\|attention mask\|generation_config"
cd listing && /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python listing.py 2>&1 | grep -v -i "warn\|Loading weights\|attention mask\|generation_config" > listing_out.txt; cd ..
