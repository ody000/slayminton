#!/bin/bash

# ============================================================
# Slayminton - OSCAR SLURM launcher
#
# Supports two modes:
#   1) train-dino : train DINO model from data/input/train
#   2) run-main   : run main loop in track-frames mode
#
# Quick usage:
#   sbatch slurm_train.sh train-dino                      # Uses train_mog_reflect (20K augmented images)
#   sbatch slurm_train.sh run-main
#
# Optional overrides at submit-time:
#   sbatch --export=ALL,EPOCHS=60,BATCH_SIZE=8,LR=5e-5 slurm_train.sh train-dino
#   sbatch --export=ALL,TRAIN_DIR=data/input/train_mog_frames slurm_train.sh train-dino    # Use baseline only
#   sbatch --export=ALL,TRAIN_DIR=data/input/train slurm_train.sh train-dino              # Use original only
#   sbatch --export=ALL,FRAMES_DIR=data/input/my_frames,FRAME_LIMIT=500 slurm_train.sh run-main
#
# Monitor:
#   myq
#   cat slurm-<jobid>.out
#   cat slurm-<jobid>.err
# ============================================================

# -------------------------
# SBATCH directives removed to allow flexible submission.
# Prefer passing resources at sbatch time, e.g.:
#   sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=02:00:00 \
#     --output=logs/dino_%j.out slurm_train.sh train-dino
# If you want defaults, set them via environment on the sbatch command line
# (see usage examples in the repo README).

## Optional runtime knobs (export via sbatch --export=ALL,VAR=val)
NUM_WORKERS="${NUM_WORKERS:-0}"
DEBUG_BATCHES="${DEBUG_BATCHES:-0}"
LOG_EVERY="${LOG_EVERY:-10}"

set -euo pipefail

# ============================================================
# MODE SELECTION
# ============================================================
# Allowed values:
#   train-dino
#   run-main
MODE="${1:-train-dino}"

# ============================================================
# PROJECT PATHS
# ============================================================
# By default we run from the submit directory.
# If you prefer a fixed path, uncomment and edit PROJECT_ROOT.
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
# PROJECT_ROOT="/users/<your_user>/slayminton"

DATA_ROOT="${PROJECT_ROOT}/data"
INPUT_ROOT="${DATA_ROOT}/input"
OUTPUT_ROOT="${DATA_ROOT}/output"

# Current training data layout in this repo.
# Options:
#   data/input/train                (original training dataset, 10K images)
#   data/input/train_mog_frames     (MOG2-masked frames, 10K images)
#   data/input/train_mog_reflect    (augmented with horizontal reflections, 20K images) [DEFAULT, RECOMMENDED]
#
# Default uses train_mog_reflect: 20K images (10K original + 10K horizontally-flipped augmented).
# Annotation files are auto-detected as "_annotations.coco.json" in each directory.
TRAIN_DIR="${TRAIN_DIR:-${INPUT_ROOT}/train_mog_reflect}"

# For run-main mode (track-frames):
# This should point to a directory of image frames (jpg/png).
# NOTE: current main.py track path expects frames, not raw video decode yet.
FRAMES_DIR="${FRAMES_DIR:-${INPUT_ROOT}/train}"

# ============================================================
# OUTPUT LAYOUT STRATEGY
# ============================================================
# We keep all cluster artifacts under one run folder:
#   data/output/oscar_runs/<mode>_<timestamp>_<jobid>/
#
# train-dino mode writes:
#   checkpoints/*.pt          (student + teacher checkpoints)
#   train_artifacts/*.npz     (training history)
#   train_artifacts/*.png     (training graphs)
#
# run-main mode writes:
#   run_artifacts/<input>_<ts>/tracking_results.json
#   run_artifacts/<input>_<ts>/rally_data.json
#   run_artifacts/<input>_<ts>/rally_statistics.json
#   run_artifacts/<input>_<ts>/rally_duration_histogram.png
#
# This is friendly for copying results off-cluster with scp/rsync.
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/oscar_runs/${MODE}_${RUN_STAMP}_${SLURM_JOB_ID}"

TRAIN_OUTPUT_DIR="${RUN_ROOT}/train_artifacts"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
MAIN_OUTPUT_DIR="${RUN_ROOT}/run_artifacts"

mkdir -p "${RUN_ROOT}" "${TRAIN_OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${MAIN_OUTPUT_DIR}"

# ============================================================
# TRAINING + TRACKING DEFAULTS
# ============================================================
# Override with --export=ALL,VAR=value at sbatch time.

# Train mode knobs
EPOCHS="${EPOCHS:-75}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-5e-4}"  # faster convergence on training plateau
WEIGHTS_NAME="${WEIGHTS_NAME:-dino_tracker.pt}"
WEIGHTS_PATH="${CHECKPOINT_DIR}/${WEIGHTS_NAME}"
# Which pretrained backbone to use (env override). Default uses DINOv2 base ViT/14.
DINOV2_MODEL="${DINOV2_MODEL:-dinov2_vitb14}"

# Run-main knobs
FPS="${FPS:-30.0}"
FRAME_LIMIT="${FRAME_LIMIT:-1200}"
RALLY_TIMEOUT_S="${RALLY_TIMEOUT_S:-0.5}"
MIN_SHUTTLE_MOTION_PX="${MIN_SHUTTLE_MOTION_PX:-2.0}"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
cd "${PROJECT_ROOT}"

# Pick ONE launcher section that matches your environment.
# Keep the others commented out.

# --- Option A: your local env shim (as in your previous scripts) ---

source "${HOME}/.local/bin/env"


# --- Option B: conda ---
# source ~/.bashrc
# conda activate <your_env_name>

# --- Option C: venv ---
# source .venv/bin/activate

# Python runner command:
# If you use uv, keep UV_RUN="uv run".
# If not, change to PYTHON_RUN="python".

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Mode:              ${MODE}"
echo "Node:              $(hostname)"
echo "Project Root:      ${PROJECT_ROOT}"
echo "Pretrained model:  ${DINOV2_MODEL}"
echo "Run Root:          ${RUN_ROOT}"
echo "Started:           $(date)"
echo "============================================"

if [[ "${MODE}" == "train-dino" ]]; then
	echo "[SLURM] Running DINO training"
	echo "[SLURM] Train dir:        ${TRAIN_DIR}"
	echo "[SLURM] Output dir:       ${TRAIN_OUTPUT_DIR}"
	echo "[SLURM] Checkpoint path:  ${WEIGHTS_PATH}"

	uv run -v python -u main.py \
		--mode train \
		--train-dir "${TRAIN_DIR}" \
		--output-dir "${TRAIN_OUTPUT_DIR}" \
		--weights "${WEIGHTS_PATH}" \
		--epochs "${EPOCHS}" \
		--batch-size "${BATCH_SIZE}" \
		--learning-rate "${LR}" \
		--pretrained-backbone "${PRETRAINED_BACKBONE:-}" \
		--dinov2-model "${DINOV2_MODEL}" \
		--num-workers "${NUM_WORKERS}" \
		--debug-batches "${DEBUG_BATCHES}" \
		--log-every "${LOG_EVERY}"

	echo "[SLURM] train-dino complete"
	echo "[SLURM] Artifacts are in: ${RUN_ROOT}"

elif [[ "${MODE}" == "run-main" ]]; then
	echo "[SLURM] Running main loop (track-frames)"
	echo "[SLURM] Frames dir:       ${FRAMES_DIR}"
	echo "[SLURM] Weights path:     ${WEIGHTS_PATH}"
	echo "[SLURM] Output dir:       ${MAIN_OUTPUT_DIR}"

	# If you trained in a previous run, point WEIGHTS_PATH to that checkpoint via:
	#   sbatch --export=ALL,WEIGHTS_PATH=/path/to/dino_tracker.pt slurm_train.sh run-main
	# If WEIGHTS_PATH doesn't exist, main.py will still run with randomly initialized model.
	uv run -v python -u main.py \
		--mode track-frames \
		--frames-dir "${FRAMES_DIR}" \
		--frame-limit "${FRAME_LIMIT}" \
		--fps "${FPS}" \
		--rally-timeout-s "${RALLY_TIMEOUT_S}" \
		--min-shuttle-motion-px "${MIN_SHUTTLE_MOTION_PX}" \
		--output-dir "${MAIN_OUTPUT_DIR}" \
		--weights "${WEIGHTS_PATH}"

	echo "[SLURM] run-main complete"
	echo "[SLURM] Artifacts are in: ${RUN_ROOT}"

else
	echo "[SLURM][ERROR] Unknown mode: ${MODE}"
	echo "[SLURM][ERROR] Use one of: train-dino | run-main"
	exit 2
fi

echo "============================================"
echo "Finished:          $(date)"
echo "============================================"

