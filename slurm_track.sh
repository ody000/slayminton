#!/bin/bash
# Simple SBATCH script to run DINO tracking (headless)
# Usage: sbatch slurm_track.sh /path/to/video.mp4 /path/to/dino_tracker.pt /path/to/dinov2_vitb14.pt /path/to/court_points.json

#SBATCH --job-name=slay_track
#SBATCH --output=slay_track-%j.out
#SBATCH --error=slay_track-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

VIDEO_PATH=${1:-data/input/match_clip.mp4}
WEIGHTS=${2:-dino_tracker.pt}
BACKBONE=${3:-}
COURT_POINTS=${4:-data/input/court_points.json}

echo "Running slayminton track job"
echo "VIDEO_PATH=${VIDEO_PATH} WEIGHTS=${WEIGHTS} BACKBONE=${BACKBONE} COURT_POINTS=${COURT_POINTS}"

# Prefer running with `uv` if available on this cluster (common wrapper on some clusters).
# Otherwise, use a provided VENV_PATH or PYTHON_BIN environment variable, or fall back to system python3.
export DINOV2_MODEL=dinov2_vitb14

PY_ARGS=(--mode track-video --video-path "${VIDEO_PATH}" --weights "${WEIGHTS}" --fps 30)
if [[ -n "${BACKBONE}" ]]; then
  PY_ARGS+=(--pretrained-backbone-path "${BACKBONE}")
fi
if [[ -n "${COURT_POINTS}" ]]; then
  PY_ARGS+=(--court-points-file "${COURT_POINTS}")
fi

run_cmd=""
if command -v uv >/dev/null 2>&1; then
  echo "Using uv to run: uv run -v python -u main.py ${PY_ARGS[*]}"
  uv run -v python -u main.py "${PY_ARGS[@]}"
  exit $?
fi

# If a specific Python binary was provided, prefer it
if [[ -n "$PYTHON_BIN" ]]; then
  run_cmd=("$PYTHON_BIN" main.py)
elif [[ -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
  # activate provided virtualenv
  source "$VENV_PATH/bin/activate"
  run_cmd=(python main.py)
else
  # fallback to system python3
  run_cmd=(python3 main.py)
fi

echo "Running: ${run_cmd[*]} ${PY_ARGS[*]}"
"${run_cmd[@]}" "${PY_ARGS[@]}"

