#!/usr/bin/env bash
# Safe overnight training queue for threading noforce runs (sequential).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/artifacts/night_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run.log"
SUMMARY="$LOGDIR/summary.txt"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] start night run" | tee -a "$LOG"
echo "[$(ts)] logdir: $LOGDIR" | tee -a "$LOG"

# Stop sweep launchers to avoid duplicate writers; resume from checkpoints.
pkill -f "run_phase2_train_sweep.py" || true
pkill -f "spine_resume_local_rgb_inpaint.sh" || true
# Stop any running train to avoid concurrent writes to same ckpt dir.
pkill -f "train_dit_min.py" || true
sleep 2

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

latest_ep() {
  python - "$1" "$2" <<'PY'
import os, re, sys
root="data/checkpoints_threading_rgb_inpaint"
group, seed = sys.argv[1], sys.argv[2]
path=os.path.join(root, group, seed)
pat=re.compile(r"spine_dit_ep(\d+)\.pth$")
latest=-1
if os.path.isdir(path):
    for fn in os.listdir(path):
        m=pat.match(fn)
        if m:
            latest=max(latest, int(m.group(1)))
print(latest)
PY
}

run_job() {
  local group="$1"
  local seed="$2"
  local dataset="$ROOT/data/spine_cito/threading/trainsets/${group}.hdf5"
  local ckpt_dir="$ROOT/data/checkpoints_threading_rgb_inpaint/${group}/${seed}"

  if [[ ! -f "$dataset" ]]; then
    echo "[$(ts)] [skip] missing dataset: $dataset" | tee -a "$LOG"
    return 0
  fi

  local ep
  ep="$(latest_ep "$group" "$seed")"
  if [[ "$ep" -ge 200 ]]; then
    echo "[$(ts)] [skip] ${group}/${seed} already ep${ep}" | tee -a "$LOG"
    return 0
  fi

  echo "[$(ts)] [start] ${group}/${seed} (latest_ep=${ep})" | tee -a "$LOG"
  python "$ROOT/train_dit_min.py" \
    --dataset "$dataset" \
    --ckpt-dir "$ckpt_dir" \
    --epochs 200 \
    --batch-size 32 \
    --horizon 16 \
    --diffusion-steps 100 \
    --force-dim 1 \
    --lr 0.0001 \
    --rgb-key agentview_rgb \
    --rgb-size 84 \
    --physics-token-dim 3 \
    --physics-mask-prob 0.5 \
    --loss-phys-weight 0.5 \
    --contact-force-threshold 2.0 \
    --force-mag-clip 50.0 \
    --seed "${seed#seed}" \
    --use-rgb \
    --use-physics-inpainting \
    --resume-latest >> "$LOG" 2>&1 || {
      echo "[$(ts)] [fail] ${group}/${seed}" | tee -a "$LOG"
      return 1
    }
  ep="$(latest_ep "$group" "$seed")"
  echo "[$(ts)] [done] ${group}/${seed} latest_ep=${ep}" | tee -a "$LOG"
}

jobs=(
  "n500_noforce seed0"
  "n500_noforce seed1"
  "n500_noforce seed2"
  "n200_noforce seed0"
  "n200_noforce seed1"
  "n200_noforce seed2"
)

for job in "${jobs[@]}"; do
  group="${job%% *}"
  seed="${job##* }"
  run_job "$group" "$seed" || true
done

{
  echo "=== summary @ $(ts) ==="
  for job in "${jobs[@]}"; do
    group="${job%% *}"
    seed="${job##* }"
    ep="$(latest_ep "$group" "$seed")"
    echo "${group}/${seed} latest_ep=${ep}"
  done
} > "$SUMMARY"

echo "[$(ts)] night run complete. Summary: $SUMMARY" | tee -a "$LOG"
