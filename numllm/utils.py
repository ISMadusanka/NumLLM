"""Small shared helpers: logging, seeding, JSON(L) IO, crash-safe state,
dtype resolution, human-readable counts."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from typing import Any, Dict, Iterable, Optional


# ------------------------------------------------------------------ logging

def setup_logging(name: str = "numllm", log_dir: Optional[str] = None,
                  level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:            # already configured this process
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ------------------------------------------------------------------ seeding

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ dtype

def resolve_dtype(name: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


# ------------------------------------------------------------------ json io

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)          # atomic on same filesystem


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlShardWriter:
    """Append-only JSONL writer that rolls to a new shard every N records.

    Shards are named <prefix>-00000.jsonl, <prefix>-00001.jsonl, ...
    Resumable: pass the shard index/record count from a saved state.
    """

    def __init__(self, out_dir: str, prefix: str, records_per_shard: int,
                 start_shard: int = 0):
        self.out_dir = out_dir
        self.prefix = prefix
        self.records_per_shard = records_per_shard
        self.shard_idx = start_shard
        self.count_in_shard = 0
        os.makedirs(out_dir, exist_ok=True)
        self._fh = None

    def _open(self):
        path = os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.jsonl")
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        if self._fh is None:
            self._open()
        if self.count_in_shard >= self.records_per_shard:
            self.close()
            self.shard_idx += 1
            self.count_in_shard = 0
            self._open()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count_in_shard += 1

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def list_shards(out_dir: str, prefix: str):
    if not os.path.isdir(out_dir):
        return []
    return sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(prefix + "-") and f.endswith(".jsonl")
    )


# ------------------------------------------------------------------ misc

def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{int(n)}"
        n /= 1000.0
    return f"{n:.2f}P"


def hard_exit(code: int = 0) -> None:
    """Exit immediately, skipping Python's interpreter finalization.

    Streaming HTTP (datasets/fsspec/aiohttp) and native tokenizer background
    threads can abort during shutdown ("Fatal Python error: PyGILState_Release
    ... must be current"). When all output is already flushed to disk, this
    sidesteps that noisy-but-harmless teardown and guarantees a clean exit code.
    """
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class StateStore:
    """Crash-safe key/value JSON state for resumable jobs."""

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                self.data = load_json(path)
            except Exception:
                self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, **kwargs) -> None:
        self.data.update(kwargs)

    def save(self) -> None:
        save_json(self.data, self.path)
