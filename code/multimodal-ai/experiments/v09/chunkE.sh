cd listing && /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python listing.py 2>&1 | grep -v -i "warn\|Loading weights\|attention mask\|generation_config" > listing_out.txt; cd ..
