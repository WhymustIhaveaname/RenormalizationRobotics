#!/bin/bash
# The whole sweep: three arms x five nominal control rates, a few chains at a time across the
# GPUs.  B is fitted with three parameters, as it is in MuJoCo, so it takes no --delay.  Each
# chain starts from the MuJoCo answer for the same grid point: the two engines agree step for
# step to float32 round-off, so the minimum is already known and the fit only has to confirm it.
cd "$(dirname "$0")"
PY=/data/mjwarp/venv/bin/python
REF=../exp1_pendulum/results
mkdir -p /data/mjwarp/logs
i=0
for arm in A B C; do
  [ "$arm" = B ] && opts="--seed $REF/flow_mujoco.json" || opts="--delay --seed $REF/flow_mujoco_delay.json"
  for fc in 10 20 50 100 200; do
    while [ "$(pgrep -u "$(id -u)" -cf 'run_single.py --arm')" -ge 4 ]; do sleep 20; done
    gpu=$(( i % 2 ))
    CUDA_VISIBLE_DEVICES=$gpu WARP_CACHE_PATH=/data/mjwarp/cache$gpu \
      setsid nohup $PY run_single.py --arm $arm --fctrl $fc $opts --only-seed \
      > /data/mjwarp/logs/${arm}_${fc}.log 2>&1 &
    i=$((i+1)); sleep 6
  done
done
wait
$PY run_single.py --merge --delay
$PY run_single.py --merge
