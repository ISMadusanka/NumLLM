"""LoRA supervised fine-tuning on the encoded NuminaMath data.

By default fine-tunes on top of the merged continual-pretrained model
(sft.start_from = cpt). Loss is masked to completion tokens only.
Checkpoints every save_steps; re-running resumes.

    python -m numllm.train.finetune --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os

from numllm.config import Config
from numllm.data.sft_dataset import DataCollatorForCausalSFT, build_sft_dataset
from numllm.models.model_utils import attach_lora, load_base_model, merge_adapter
from numllm.tokenizer_utils import get_extended_tokenizer, load_plain_tokenizer
from numllm.train.common import bf16_fp16, latest_checkpoint, make_loss_logger
from numllm.utils import human, save_json, set_seed, setup_logging


def _resolve_base(cfg, logger) -> str:
    if cfg.sft.start_from == "cpt":
        merged = os.path.join(cfg.paths.cpt_out, "merged")
        if not os.path.isdir(merged):
            raise FileNotFoundError(
                f"start_from=cpt but no merged CPT model at {merged}. "
                f"Run continual_pretrain (with cpt.merge_after=true) first, "
                f"or set --set sft.start_from=base")
        return merged
    return cfg.model.base_model


def run(cfg: Config, logger):
    from transformers import Trainer, TrainingArguments

    set_seed(cfg.sft.seed)
    base_path = _resolve_base(cfg, logger)
    logger.info(f"Fine-tuning on top of: {base_path}")

    if cfg.encoding.enabled:
        tok, _ = get_extended_tokenizer(cfg)
        tok_dir = cfg.paths.tokenizer
    else:
        tok = load_plain_tokenizer(cfg)
        tok_dir = base_path

    # If we start from the raw base model but still use encoding, the new-token
    # embeddings are untrained here, so train them in full.
    if cfg.sft.start_from == "base" and cfg.encoding.enabled:
        modules_to_save = cfg.lora.cpt_modules_to_save
    else:
        modules_to_save = cfg.lora.sft_modules_to_save

    model = load_base_model(cfg, model_path=base_path, tokenizer=tok, for_training=True)
    model = attach_lora(model, cfg, modules_to_save)

    examples = build_sft_dataset(cfg.paths.finetune_data, tok, cfg.sft.max_seq_len)
    logger.info(f"SFT dataset: {human(len(examples))} examples")
    collator = DataCollatorForCausalSFT(tokenizer=tok)

    bf16, fp16 = bf16_fp16(cfg)
    ta_kwargs = dict(
        output_dir=cfg.paths.sft_out,
        per_device_train_batch_size=cfg.sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.sft.grad_accum,
        learning_rate=cfg.sft.lr,
        weight_decay=cfg.sft.weight_decay,
        warmup_ratio=cfg.sft.warmup_ratio,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        logging_steps=cfg.sft.logging_steps,
        save_steps=cfg.sft.save_steps,
        save_total_limit=cfg.sft.save_total_limit,
        bf16=bf16, fp16=fp16,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
        logging_dir=os.path.join(cfg.paths.logs, "sft_tb"),
        seed=cfg.sft.seed,
    )
    if cfg.sft.max_steps and cfg.sft.max_steps > 0:
        ta_kwargs["max_steps"] = cfg.sft.max_steps
    else:
        ta_kwargs["num_train_epochs"] = cfg.sft.num_train_epochs

    trainer = Trainer(model=model, args=TrainingArguments(**ta_kwargs),
                      train_dataset=examples, data_collator=collator,
                      callbacks=[make_loss_logger(logger)])

    ckpt = latest_checkpoint(cfg.paths.sft_out)
    if ckpt:
        logger.info(f"Resuming from checkpoint: {ckpt}")
    logger.info("Starting fine-tuning ...")
    result = trainer.train(resume_from_checkpoint=ckpt)

    trainer.save_model(cfg.paths.sft_out)
    tok.save_pretrained(cfg.paths.sft_out)
    save_json({k: v for k, v in result.metrics.items()},
              os.path.join(cfg.paths.sft_out, "train_metrics.json"))
    logger.info(f"Saved SFT adapter -> {cfg.paths.sft_out}")

    if cfg.sft.merge_after:
        import gc
        import torch
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merged = os.path.join(cfg.paths.sft_out, "merged")
        logger.info(f"Merging adapter into full model -> {merged}")
        merge_adapter(cfg, cfg.paths.sft_out, merged,
                      base_path=base_path, tokenizer_dir=tok_dir)
        logger.info("Merge complete.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Supervised fine-tuning (LoRA)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    logger = setup_logging("finetune", cfg.paths.logs)
    run(cfg, logger)


if __name__ == "__main__":
    main()
