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
# SLURM resources
# -------------------------
# Tune these for your account/queue policy.
# If jobs stay pending too long, reduce time/mem/gpu requests.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J slayminton
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

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
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"  # default LR for fine-tuning
WEIGHTS_NAME="${WEIGHTS_NAME:-dino_tracker.pt}"
WEIGHTS_PATH="${CHECKPOINT_DIR}/${WEIGHTS_NAME}"
# Which pretrained backbone to use (env override). Default uses DINOv2 base ViT/14.
DINOV2_MODEL="${DINOV2_MODEL:-dinov2_vitb14}"
# LoRA env knobs (enable by exporting USE_LORA=1 at sbatch time)
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
USE_DDP="${USE_DDP:-0}"
# How many processes per node to pass to torchrun when USE_DDP=1
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# AMP toggle (no-op unless main.py/train_dino support enabled)
USE_AMP="${USE_AMP:-0}"
# Default number of dataloader workers for training
NUM_WORKERS="${NUM_WORKERS:-4}"

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

# Prefer activating a project venv if present, otherwise fall back to user env shim.
# This ensures `torch`/`torchrun` are available in the non-interactive SLURM job.
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
	echo "[SLURM] Activating project venv: ${PROJECT_ROOT}/.venv"
	# shellcheck disable=SC1091
	source "${PROJECT_ROOT}/.venv/bin/activate"
	VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
	VENV_TORCHRUN="${PROJECT_ROOT}/.venv/bin/torchrun"
elif [[ -f "${HOME}/.local/bin/env" ]]; then
	echo "[SLURM] Sourcing user env shim: ${HOME}/.local/bin/env"
	# shellcheck disable=SC1091
	source "${HOME}/.local/bin/env"
	VENV_PY="$(which python || true)"
	VENV_TORCHRUN="$(which torchrun || true)"
else
	VENV_PY="$(which python || true)"
	VENV_TORCHRUN="$(which torchrun || true)"
fi

# Helpful PyTorch allocator settings to reduce fragmentation (can be overridden via sbatch --export)
: ${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True,max_split_size_mb:128}
export PYTORCH_CUDA_ALLOC_CONF

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

# Canonicalize relative dataset paths to absolute paths under PROJECT_ROOT
if [[ "${TRAIN_DIR}" != /* ]]; then
	TRAIN_DIR="${PROJECT_ROOT}/${TRAIN_DIR#./}"
fi
if [[ "${FRAMES_DIR}" != /* ]]; then
	FRAMES_DIR="${PROJECT_ROOT}/${FRAMES_DIR#./}"
fi

# Quick data existence check (helps debug missing files on compute node)
if [[ ! -d "${TRAIN_DIR}" ]]; then
	echo "[SLURM][WARNING] TRAIN_DIR does not exist on node: ${TRAIN_DIR}"
else
	echo "[SLURM] TRAIN_DIR exists: ${TRAIN_DIR} (showing up to 5 files)"
	ls -1 "${TRAIN_DIR}" | head -n 5 || true
fi

if [[ "${MODE}" == "train-dino" ]]; then
	echo "[SLURM] Running DINO training"
	echo "[SLURM] Train dir:        ${TRAIN_DIR}"
	echo "[SLURM] Output dir:       ${TRAIN_OUTPUT_DIR}"
	echo "[SLURM] Checkpoint path:  ${WEIGHTS_PATH}"

	# Build base command and append LoRA flags conditionally
	if [[ "${USE_LORA}" == "1" ]]; then
		UV_CMD_LORA=" --use-lora --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA}"
	else
		UV_CMD_LORA=" --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA}"
	fi

	# Compose the runner: use torchrun when USE_DDP=1, otherwise run via uv/python
	if [[ "${USE_DDP}" == "1" ]]; then
		echo "[SLURM] USE_DDP=1 -> launching with torchrun (nproc_per_node=${NPROC_PER_NODE})"
		# Prefer torchrun from the activated venv, fall back to venv python -m torch.distributed.run, then system torchrun
		if [[ -n "${VENV_TORCHRUN:-}" && -x "${VENV_TORCHRUN}" ]]; then
			RUNNER_CMD=("${VENV_TORCHRUN}" --nproc_per_node ${NPROC_PER_NODE} --standalone)
		elif [[ -n "${VENV_PY:-}" && -x "${VENV_PY}" ]]; then
			RUNNER_CMD=("${VENV_PY}" -m torch.distributed.run --nproc_per_node ${NPROC_PER_NODE} --standalone)
		else
			RUNNER_CMD=(torchrun --nproc_per_node ${NPROC_PER_NODE} --standalone)
		fi
		# torchrun will set LOCAL_RANK per process; pass --use-ddp to main
		EXTRA_DDP_FLAG=" --use-ddp"
	else
		RUNNER_CMD=(uv run -v python -u)
		EXTRA_DDP_FLAG=""
	fi

	# Final training command
	"${RUNNER_CMD[@]}" main.py \
		--mode train \
		--train-dir "${TRAIN_DIR}" \
		--output-dir "${TRAIN_OUTPUT_DIR}" \
		--weights "${WEIGHTS_PATH}" \
		--epochs "${EPOCHS}" \
		--batch-size "${BATCH_SIZE}" \
		--learning-rate "${LR}" ${UV_CMD_LORA} ${EXTRA_DDP_FLAG}

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

## Optional runtime knobs (export via sbatch --export=ALL,VAR=val)
NUM_WORKERS="${NUM_WORKERS:-0}"
DEBUG_BATCHES="${DEBUG_BATCHES:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
DEBUG_MODE="${DEBUG_MODE:-0}"

# If DEBUG_MODE is enabled (export DEBUG_MODE=1), reduce run sizes for quick debug.
if [[ "${DEBUG_MODE}" == "1" ]]; then
  echo "[SLURM] DEBUG_MODE=1 -> applying debug defaults: EPOCHS=1, BATCH_SIZE=4, NUM_WORKERS=0, DEBUG_BATCHES=1, LOG_EVERY=5"
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  NUM_WORKERS="0"
  DEBUG_BATCHES="1"
  LOG_EVERY="5"
fi

