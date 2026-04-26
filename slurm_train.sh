#!/bin/bash

# ============================================================
# Slayminton - OSCAR SLURM launcher
#
# Supports two modes:
#   1) train-dino : train DINO model from data/input/train
#   2) run-main   : run main loop in track-frames mode
#
# Quick usage:
#   sbatch slurm_train.sh train-dino
#   sbatch slurm_train.sh run-main
#
# Optional overrides at submit-time:
#   sbatch --export=ALL,EPOCHS=60,BATCH_SIZE=8,LR=5e-5 slurm_train.sh train-dino
#   sbatch --export=ALL,FRAMES_DIR=data/input/my_frames,FRAME_LIMIT=500 slurm_train.sh run-main
#
# Monitor:
#   myq
#   cat slurm-<jobid>.out
#   cat slurm-<jobid>.err
# ============================================================

# -------------------------
# SLURM resources
# -------------------------
# Tune these for your account/queue policy.
# If jobs stay pending too long, reduce time/mem/gpu requests.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -J slayminton
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

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
TRAIN_DIR="${INPUT_ROOT}/train"
ANNOTATIONS_FILE="${TRAIN_DIR}/_annotations.coco.json"

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
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
WEIGHTS_NAME="${WEIGHTS_NAME:-dino_tracker.pt}"
WEIGHTS_PATH="${CHECKPOINT_DIR}/${WEIGHTS_NAME}"

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
echo "Run Root:          ${RUN_ROOT}"
echo "Started:           $(date)"
echo "============================================"

if [[ "${MODE}" == "train-dino" ]]; then
	echo "[SLURM] Running DINO training"
	echo "[SLURM] Train dir:        ${TRAIN_DIR}"
	echo "[SLURM] Annotations:      ${ANNOTATIONS_FILE}"
	echo "[SLURM] Output dir:       ${TRAIN_OUTPUT_DIR}"
	echo "[SLURM] Checkpoint path:  ${WEIGHTS_PATH}"

	uv run -v python -u main.py \
		--mode train \
		--train-dir "${TRAIN_DIR}" \
		--annotations "${ANNOTATIONS_FILE}" \
		--output-dir "${TRAIN_OUTPUT_DIR}" \
		--weights "${WEIGHTS_PATH}" \
		--epochs "${EPOCHS}" \
		--batch-size "${BATCH_SIZE}" \
		--learning-rate "${LR}"

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

