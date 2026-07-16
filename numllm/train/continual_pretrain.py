"""Continual pre-training of the base model on encoded FineWeb, with LoRA.

New numeric tokens' embeddings are trained in full (lora.cpt_modules_to_save).
Checkpoints every save_steps; re-running resumes from the last checkpoint.

    python -m numllm.train.continual_pretrain --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os

from numllm.config import Config
from numllm.data.lm_dataset import PackedJsonlDataset, cpt_data_collator
from numllm.models.model_utils import attach_lora, load_base_model, merge_adapter
from numllm.train.common import bf16_fp16, latest_checkpoint, make_loss_logger
from numllm.tokenizer_utils import get_extended_tokenizer
from numllm.utils import human, load_json, save_json, set_seed, setup_logging


def _derive_max_steps(cfg, logger) -> int:
    if cfg.cpt.max_steps and cfg.cpt.max_steps > 0:
        return cfg.cpt.max_steps
    manifest = os.path.join(cfg.paths.pretrain_data, "manifest.json")
    total = cfg.pretrain_data.target_tokens
    if os.path.exists(manifest):
        written = load_json(manifest).get("tokens_written", total)
        # Cap at target_tokens so lowering it (e.g. 5B -> 1B) shortens training
        # even when more tokens were already preprocessed (no re-preprocess needed).
        total = min(written, cfg.pretrain_data.target_tokens)
    world = int(os.environ.get("WORLD_SIZE", 1))
    tokens_per_step = (cfg.cpt.per_device_batch_size * cfg.cpt.grad_accum
                       * world * cfg.cpt.block_size)
    steps = max(1, total // tokens_per_step)
    logger.info(f"Derived max_steps={steps} from {human(total)} tokens "
                f"({human(tokens_per_step)} tokens/step)")
    return steps


def run(cfg: Config, logger):
    from transformers import Trainer, TrainingArguments

    set_seed(cfg.cpt.seed)
    tok, added = get_extended_tokenizer(cfg)
    logger.info(f"Tokenizer vocab={len(tok)} (numeric tokens added earlier: {added})")

    model = load_base_model(cfg, tokenizer=tok, for_training=True)
    model = attach_lora(model, cfg, cfg.lora.cpt_modules_to_save)

    train_ds = PackedJsonlDataset(cfg.paths.pretrain_data, "pretrain", tok,
                                  cfg.cpt.block_size, seed=cfg.cpt.seed)
    collator = cpt_data_collator

    bf16, fp16 = bf16_fp16(cfg)
    max_steps = _derive_max_steps(cfg, logger)

    args = TrainingArguments(
        output_dir=cfg.paths.cpt_out,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.cpt.per_device_batch_size,
        gradient_accumulation_steps=cfg.cpt.grad_accum,
        learning_rate=cfg.cpt.lr,
        weight_decay=cfg.cpt.weight_decay,
        warmup_ratio=cfg.cpt.warmup_ratio,
        lr_scheduler_type=cfg.cpt.lr_scheduler_type,
        logging_steps=cfg.cpt.logging_steps,
        save_steps=cfg.cpt.save_steps,
        save_total_limit=cfg.cpt.save_total_limit,
        bf16=bf16, fp16=fp16,
        gradient_checkpointing=cfg.cpt.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        report_to=[],
        logging_dir=os.path.join(cfg.paths.logs, "cpt_tb"),
        seed=cfg.cpt.seed,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=collator, callbacks=[make_loss_logger(logger)])

    ckpt = latest_checkpoint(cfg.paths.cpt_out)
    if ckpt:
        logger.info(f"Resuming from checkpoint: {ckpt}")
    logger.info("Starting continual pre-training ...")
    result = trainer.train(resume_from_checkpoint=ckpt)

    trainer.save_model(cfg.paths.cpt_out)          # LoRA adapter + trained embeddings
    tok.save_pretrained(cfg.paths.cpt_out)
    save_json({k: v for k, v in result.metrics.items()},
              os.path.join(cfg.paths.cpt_out, "train_metrics.json"))
    logger.info(f"Saved CPT adapter -> {cfg.paths.cpt_out}")

    if cfg.cpt.merge_after:
        import gc
        import torch
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merged = os.path.join(cfg.paths.cpt_out, "merged")
        logger.info(f"Merging adapter into full model -> {merged}")
        merge_adapter(cfg, cfg.paths.cpt_out, merged,
                      base_path=cfg.model.base_model, tokenizer_dir=cfg.paths.tokenizer)
        logger.info("Merge complete.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Continual pre-training (LoRA)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    logger = setup_logging("continual_pretrain", cfg.paths.logs)
    run(cfg, logger)


if __name__ == "__main__":
    main()
