"""Shared training utilities: dtype flags, loss logging, checkpoint discovery."""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def bf16_fp16(cfg):
    d = cfg.model.dtype.lower()
    return (d in ("bfloat16", "bf16")), (d in ("float16", "fp16"))


def latest_checkpoint(output_dir: str):
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:
        return None
    if output_dir and os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def make_loss_logger(logger):
    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            keys = ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch")
            parts = [f"{k}={logs[k]:.4g}" if isinstance(logs[k], float) else f"{k}={logs[k]}"
                     for k in keys if k in logs]
            if parts:
                logger.info(f"step {state.global_step}: " + " ".join(parts))

    return LossLogger()
