"""Supervised fine-tuning dataset (prompt/completion with loss masking).

Loss is computed only on completion tokens; prompt tokens get label -100.
The dataset (~10M tokens) is small, so we build it in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from numllm.utils import iter_jsonl, list_shards


def build_sft_dataset(out_dir: str, tokenizer, max_seq_len: int, prefix: str = "finetune"):
    """Return an in-memory list of {input_ids, labels} examples."""
    shards = list_shards(out_dir, prefix)
    if not shards:
        raise FileNotFoundError(
            f"No '{prefix}-*.jsonl' shards in {out_dir}. Run preprocessing first.")
    eos = tokenizer.eos_token_id
    examples = []
    for shard in shards:
        for rec in iter_jsonl(shard):
            p_ids = tokenizer(rec["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(rec["completion"], add_special_tokens=False)["input_ids"]
            c_ids = c_ids[: max_seq_len - 1] + [eos]          # keep room for EOS
            max_prompt = max_seq_len - len(c_ids)
            if max_prompt <= 0:
                p_ids = []
            else:
                p_ids = p_ids[-max_prompt:]                   # keep the problem tail
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + list(c_ids)
            examples.append({"input_ids": input_ids, "labels": labels})
    return examples


@dataclass
class DataCollatorForCausalSFT:
    tokenizer: object
    label_pad_token_id: int = -100

    def __call__(self, features: List[dict]) -> dict:
        import torch
        pad_id = self.tokenizer.pad_token_id
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [self.label_pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
