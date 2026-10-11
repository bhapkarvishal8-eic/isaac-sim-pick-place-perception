#!/usr/bin/env bash
# Headless, no-recording runs over several random seeds (box positions/yaws).
ISAAC=/home/vishal/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64
for s in "$@"; do
  OUT=/home/vishal/poc/logs/final_demo/robust_seed$s; rm -rf $OUT
  ( cd $ISAAC && DEMO_HEADLESS=1 DEMO_RECORD=0 DEMO_SEED=$s DEMO_OUT=$OUT timeout 1200 ./kit/kit apps/isaacsim.exp.base.python.kit \
      --no-window --exec /home/vishal/poc/stage1_perception_arm/final_demo/pick_place_final.py > $OUT.kit.log 2>&1 )
  echo "seed $s: $(grep -E 'DEMO COMPLETE|DEMO FAILED' $OUT/demo_log.txt | cut -c1-200)"
done
