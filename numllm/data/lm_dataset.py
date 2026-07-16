"""Packed language-modeling dataset for continual pretraining.

Reads the preprocessed JSONL shards, tokenizes on the fly, and yields
contiguous `block_size` chunks. Streaming keeps memory flat even at billions of
tokens. Shards/chunks are shuffled with a small buffer for better mixing,
and shard files are split across DataLoader workers.
"""

from __future__ import annotations

import random
from typing import List

from torch.utils.data import IterableDataset, get_worker_info

from numllm.utils import iter_jsonl, list_shards


class PackedJsonlDataset(IterableDataset):
    def __init__(self, out_dir: str, prefix: str, tokenizer, block_size: int,
                 text_field: str = "text", shuffle_buffer: int = 2000,
                 seed: int = 1234):
        super().__init__()
        self.shards = list_shards(out_dir, prefix)
        if not self.shards:
            raise FileNotFoundError(
                f"No '{prefix}-*.jsonl' shards in {out_dir}. Run preprocessing first.")
        self.tok = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.eos_id = tokenizer.eos_token_id

    def _my_shards(self) -> List[str]:
        # Stripe shards across DataLoader workers only. DDP/rank partitioning is
        # handled by HF Trainer's IterableDatasetShard wrapper, so we must not
        # also shard by rank here (that would double-shard).
        info = get_worker_info()
        n_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0

        shards = list(self.shards)
        random.Random(self.seed).shuffle(shards)              # identical order in every worker
        return shards[worker_id::n_workers]

    def __iter__(self):
        info = get_worker_info()
        rng = random.Random(self.seed + 7919 * (info.id if info else 0))
        token_buf: List[int] = []
        chunk_buf = []

        def emit(chunk):
            chunk_buf.append({"input_ids": chunk})
            if len(chunk_buf) >= self.shuffle_buffer:
                rng.shuffle(chunk_buf)
                while chunk_buf:
                    yield chunk_buf.pop()

        for shard in self._my_shards():
            for rec in iter_jsonl(shard):
                ids = self.tok(rec[self.text_field], add_special_tokens=False)["input_ids"]
                ids.append(self.eos_id)
                token_buf.extend(ids)
                while len(token_buf) >= self.block_size:
                    chunk = token_buf[:self.block_size]
                    del token_buf[:self.block_size]
                    yield from emit(chunk)
        rng.shuffle(chunk_buf)
        while chunk_buf:
            yield chunk_buf.pop()


def cpt_data_collator(features):
    """Stack equal-length packed blocks; labels = input_ids (train on every
    token, including the EOS separators). No padding needed."""
    import torch
    ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    return {"input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone()}
