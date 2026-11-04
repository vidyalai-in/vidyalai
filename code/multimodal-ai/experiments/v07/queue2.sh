#!/bin/bash
# After train-rf-unet (already waiting on the lock) finishes: three merged locked chunks.
cd /home/amit/.cache/vidyalai-scratch/v07
P=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
L="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 bash gpu.sh"
memok() { until [ $(free -m | awk '/Mem:/{print $7}') -ge 4500 ]; do sleep 30; done; }
until grep -q "train-rf-unet" gpu_minutes.log; do sleep 15; done
memok; echo "$(date +%T) X"
$L X-dit-reflow bash -c "timeout 420 $P train.py --name rf-dit --model dit --obj flow --steps 4000 --snap 500 --minutes 6.8; timeout 240 $P evalgen.py pairs --ckpt ckpt/rf-unet-snap-4000.pt --model unet --steps 32 --n 20000 --pairs ckpt/pairs.pt && timeout 200 $P train.py --name reflow-unet --model unet --obj flow --init ckpt/rf-unet-snap-4000.pt --pairs ckpt/pairs.pt --lr 1e-4 --steps 1500 --snap 500 --minutes 3" > runs/X.log 2>&1
memok; echo "$(date +%T) Y"
S=1,2,4,8,16,32,100
$L Y-sweep bash -c "timeout 400 $P evalgen.py steps --ckpt ckpt/ddpm-unet-snap-4000.pt --kind ddpm --model unet --steps $S --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-ddpm.npz; timeout 400 $P evalgen.py steps --ckpt ckpt/rf-unet-snap-4000.pt --kind flow --model unet --steps $S --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-rf.npz; timeout 150 $P evalgen.py steps --ckpt ckpt/reflow-unet-snap-1500.pt --kind flow --model unet --steps 1,2,4,8,16,32 --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-reflow.npz" > runs/Y.log 2>&1
memok; echo "$(date +%T) Z"
$L Z-snaps-straight bash -c "for m in 'rf-unet flow unet' 'rf-dit flow dit' 'ddpm-unet ddpm unet'; do set -- \$m; timeout 150 $P evalgen.py snaps --glob \"ckpt/\$1-snap-*.pt\" --kind \$2 --model \$3 --steps 16 --n 2000 --out runs/snaps.jsonl; done; for m in 'ddpm-unet ddpm unet 4000' 'rf-unet flow unet 4000' 'reflow-unet flow unet 1500' 'rf-dit flow dit 4000'; do set -- \$m; timeout 120 $P evalgen.py straight --ckpt ckpt/\$1-snap-\$4.pt --kind \$2 --model \$3 --steps 100 --n 1000 --tag \$1 --out runs/straight.jsonl --save runs/traj-\$1.npz; done; timeout 150 $P timing.py" > runs/Z.log 2>&1
echo "$(date +%T) all done"
