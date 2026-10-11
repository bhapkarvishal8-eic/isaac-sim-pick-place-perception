#!/usr/bin/env bash
# Final perception-driven pick-and-place demo (Isaac Sim 6.0.1).
#
#   ./run_final_demo.sh            live: opens the Isaac Sim window and runs the demo (window stays open)
#   ./run_final_demo.sh --video    headless: runs the demo and writes demo.mp4
#   options: --seed N  (different random box positions/yaws, default 7)
#
# Outputs: /home/vishal/poc/logs/final_demo/<mode>_seed<N>/
#   demo_log.txt, summary.json, perception/*.png (camera frames + detections), demo.mp4 (--video)
set -euo pipefail
ISAAC=/home/vishal/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64
SCRIPT=/home/vishal/poc/stage1_perception_arm/final_demo/pick_place_final.py

MODE=live
SEED=7
while [ $# -gt 0 ]; do
  case "$1" in
    --video) MODE=video ;;
    --seed) SEED="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

OUT=/home/vishal/poc/logs/final_demo/${MODE}_seed${SEED}
rm -rf "$OUT"; mkdir -p "$OUT"
# don't fight another sim for the GPU
pkill -f "kit/ki[t] apps/isaacsim" 2>/dev/null || true
sleep 1

cd "$ISAAC"
if [ "$MODE" = video ]; then
  echo "[demo] headless run, seed $SEED -> $OUT"
  DEMO_HEADLESS=1 DEMO_RECORD=1 DEMO_SEED=$SEED DEMO_OUT=$OUT \
    ./kit/kit apps/isaacsim.exp.base.python.kit --no-window --exec "$SCRIPT" > "$OUT/kit.log" 2>&1
  grep -E "DEMO COMPLETE|DEMO FAILED" "$OUT/demo_log.txt" || true
  echo "[demo] encoding video"
  ffmpeg -loglevel error -y -framerate 30 -i "$OUT/frames/f_%06d.jpg" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/demo.mp4"
  rm -rf "$OUT/frames"
  echo "[demo] video: $OUT/demo.mp4"
else
  echo "[demo] live run in the Isaac Sim window, seed $SEED -> $OUT"
  echo "[demo] progress: tail -f $OUT/demo_log.txt"
  DEMO_HEADLESS=0 DEMO_RECORD=0 DEMO_SEED=$SEED DEMO_OUT=$OUT \
    ./kit/kit apps/isaacsim.exp.full.kit --exec "$SCRIPT" > "$OUT/kit.log" 2>&1
fi
