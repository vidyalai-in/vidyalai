until grep -q "download done" /home/amit/.cache/vidyalai-scratch/t03/logs/sample_download.log; do sleep 15; done; tail -12 /home/amit/.cache/vidyalai-scratch/t03/logs/sample_download.log
