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

module purge
module load anaconda/2023.11
source activate slayminton || source /path/to/venv/bin/activate

# Ensure dinov2 loader uses the right model name
export DINOV2_MODEL=dinov2_vitb14

PY_ARGS=(--mode track-video --video-path "${VIDEO_PATH}" --weights "${WEIGHTS}" --fps 30)
if [[ -n "${BACKBONE}" ]]; then
  PY_ARGS+=(--pretrained-backbone-path "${BACKBONE}")
fi
if [[ -n "${COURT_POINTS}" ]]; then
  PY_ARGS+=(--court-points-file "${COURT_POINTS}")
fi

python3 main.py "${PY_ARGS[@]}"
