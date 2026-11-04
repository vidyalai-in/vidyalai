#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v07
P=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
L="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 bash gpu.sh"
memok() { until [ $(free -m | awk '/Mem:/{print $7}') -ge 4500 ]; do sleep 30; done; }
S=1,2,4,8,16,32,100
memok; echo "$(date +%T) E1"
$L E1-sweep bash -c "timeout 840 $P evalgen.py steps --ckpt ckpt/ddpm-unet-snap-4000.pt --kind ddpm --model unet --steps $S --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-ddpm.npz && timeout 840 $P evalgen.py steps --ckpt ckpt/rf-unet-snap-4000.pt --kind flow --model unet --steps $S --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-rf.npz" > runs/E1.log 2>&1
memok; echo "$(date +%T) E23"
$L E23-snaps-reflow bash -c "timeout 840 $P evalgen.py snaps --glob 'ckpt/rf-unet-snap-*.pt' --kind flow --model unet --steps 16 --n 2000 --out runs/snaps.jsonl && timeout 840 $P evalgen.py snaps --glob 'ckpt/rf-dit-snap-*.pt' --kind flow --model dit --steps 16 --n 2000 --out runs/snaps.jsonl && timeout 840 $P evalgen.py snaps --glob 'ckpt/ddpm-unet-snap-*.pt' --kind ddpm --model unet --steps 16 --n 2000 --out runs/snaps.jsonl; timeout 400 $P evalgen.py pairs --ckpt ckpt/rf-unet-snap-4000.pt --model unet --steps 32 --n 20000 --pairs ckpt/pairs.pt && timeout 400 $P train.py --name reflow-unet --model unet --obj flow --init ckpt/rf-unet-snap-4000.pt --pairs ckpt/pairs.pt --lr 1e-4 --steps 1500 --snap 500 --minutes 6" > runs/E23.log 2>&1
memok; echo "$(date +%T) E4"
$L E4-final bash -c "timeout 400 $P evalgen.py steps --ckpt ckpt/reflow-unet-snap-1500.pt --kind flow --model unet --steps 1,2,4,8,16,32 --seeds 0,1,2 --n 2000 --out runs/sweep.jsonl --save runs/grid-reflow.npz; for m in 'ddpm-unet ddpm unet' 'rf-unet flow unet' 'reflow-unet flow unet' 'rf-dit flow dit'; do set -- \$m; st=4000; [ \$1 = reflow-unet ] && st=1500; timeout 200 $P evalgen.py straight --ckpt ckpt/\$1-snap-\$st.pt --kind \$2 --model \$3 --steps 100 --n 1000 --tag \$1 --out runs/straight.jsonl --save runs/traj-\$1.npz; done; timeout 200 $P timing.py" > runs/E4.log 2>&1
echo "$(date +%T) all done"
