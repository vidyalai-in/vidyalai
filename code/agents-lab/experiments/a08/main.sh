#!/bin/bash
# Main run: all Haiku arms x 3 repeats, then Sonnet 5.5 headline (full x 3, title_only x 2). Resumable.
cd /home/amit/.cache/vidyalai-scratch/a08
RUN="systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python run.py"
ARMS=full,no_tests,rewrite,no_search,map,map_no_search,title_only
$RUN --arms $ARMS --repeats 3
$RUN --arms full --repeats 3 --model claude-sonnet-5-5 --max-cost 2.0
$RUN --arms title_only --repeats 2 --model claude-sonnet-5-5 --max-cost 2.0
echo MAIN DONE
