#!/bin/bash
# =============================================================================
# SLURM job script — NumLLM: preprocess -> preprocess -> continual pre-train.
#
# Runs the three pipeline steps IN ORDER, in one GPU job:
#   1. numllm.data.preprocess_pretrain    (FineWeb  -> encoded pretrain shards)
#   2. numllm.data.preprocess_finetune    (NuminaMath -> encoded SFT shards)
#   3. numllm.train.continual_pretrain    (CPT of Qwen2.5-3B on the shards)
#
# Submit from the repo root on deie01:
#     mkdir -p logs/slurm && sbatch train.sh
#
# Track it:            squeue -u $USER
# Watch the log:       tail -f logs/slurm/<job-name>_<jobid>.out
# Cancel it:           scancel <jobid>
#
# Each step checkpoints and RESUMES after a crash/timeout — just resubmit.
# Use a different config with:  CONFIG=configs/myconfig.yaml sbatch train.sh
# =============================================================================

#SBATCH --job-name=49_Dinuth_numllm        # CHANGE to GroupNo_StudentName_AnyOther
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                        # one GPU
#SBATCH --cpus-per-task=8                   # preprocessing is CPU/IO heavy
#SBATCH --mem=32G                           # 3B CPT + dataset streaming
#SBATCH --time=48:00:00                     # safety cap (partition limit is infinite)
#SBATCH --output=logs/slurm/%x_%j.out       # %x = job-name, %j = job id
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# --- Run from the directory the job was submitted from --------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

# --- Activate the project's venv (built by setup_env.sh) ------------------
ENV_DIR="${NUMLLM_ENV:-$SLURM_SUBMIT_DIR/venv}"
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "[env] activating project venv: $ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
else
    echo "ERROR: venv not found at $ENV_DIR. Run setup_env.sh once first." >&2
    exit 1
fi

# Fail early if torch isn't importable.
python -c "import torch" 2>/dev/null || {
    echo "ERROR: torch not importable. Build the venv:  bash setup_env.sh" >&2
    exit 1
}

# --- Provenance / sanity in the log ---------------------------------------
CONFIG="${CONFIG:-configs/default.yaml}"
echo "=================================================================="
echo "Job          : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node         : $(hostname)"
echo "GPUs (SLURM) : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config       : $CONFIG"
echo "Started      : $(date)"
echo "=================================================================="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# --- Pipeline (in order; set -e stops the job if any step fails) -----------
echo ""
echo "########## [1/3] preprocess_pretrain :: $(date) ##########"
srun python -m numllm.data.preprocess_pretrain --config "$CONFIG"

echo ""
echo "########## [2/3] preprocess_finetune :: $(date) ##########"
srun python -m numllm.data.preprocess_finetune --config "$CONFIG"

echo ""
echo "########## [3/3] continual_pretrain :: $(date) ##########"
srun python -m numllm.train.continual_pretrain --config "$CONFIG"

echo ""
echo "All steps finished: $(date)"
