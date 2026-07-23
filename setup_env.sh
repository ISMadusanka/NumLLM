#!/bin/bash
# =============================================================================
# One-time environment setup for NumLLM on the GPU cluster (deie01).
#
# Run ONCE on deie01, from inside the copied project directory:
#
#     ssh student@10.50.20.197                      # deie01
#     cd ~/NumLLM                                   # the copied project
#     srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
#     bash setup_env.sh
#     exit                                          # release the allocation
#
# Creates ./venv INSIDE the project — the exact path train.sh activates.
# =============================================================================
set -euo pipefail

# Create the venv inside the project by default (matches train.sh).
# --system-site-packages reuses the shared ai_env where possible.
ENV_DIR="${NUMLLM_ENV:-$PWD/venv}"

echo "[setup] Creating virtual environment at: $ENV_DIR"
python3 -m venv --system-site-packages "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "[setup] Upgrading pip"
pip install --upgrade pip

# RTX 5090 / Blackwell (sm_120) needs a CUDA 12.8 PyTorch build — install it
# FIRST so the plain `torch` in requirements.txt is already satisfied.
echo "[setup] Installing CUDA 12.8 PyTorch"
pip install --index-url https://download.pytorch.org/whl/cu128 torch

echo "[setup] Installing project requirements"
pip install -r requirements.txt

# transformers' apply_chat_template needs jinja2>=3.1.0; a --system-site-packages
# venv can inherit an older system jinja2, so force the upgrade into the venv.
echo "[setup] Ensuring jinja2 >= 3.1.0"
pip install "jinja2>=3.1.0"

echo "[setup] Sanity check: torch + CUDA visibility"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[setup] Done. In job scripts, activate with:  source $ENV_DIR/bin/activate"
