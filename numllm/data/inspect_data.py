"""Inspect preprocessed data.

    python -m numllm.data.inspect_data --split pretrain --n 5
    python -m numllm.data.inspect_data --split finetune --n 5

Prints raw -> encoded samples, decodes the encoded numbers back to verify the
round-trip, and shows the manifest / progress summary.
"""

from __future__ import annotations

import argparse
import json
import os

from numllm.config import Config
from numllm.encoding_utils import NumberCodec

_RULE = "=" * 78


def _print_manifest(out_dir: str):
    for fname in ("manifest.json", "state.json"):
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            d.pop("hf_state", None)
            print(f"\n[{fname}] {json.dumps(d, indent=2)}")


def _preview(out_dir: str, n: int):
    path = os.path.join(out_dir, "preview.jsonl")
    if not os.path.exists(path):
        print(f"(no preview.jsonl in {out_dir} yet — run preprocessing first)")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def inspect_pretrain(cfg: Config, n: int):
    out_dir = cfg.paths.pretrain_data
    codec = NumberCodec.from_config(cfg)
    print(f"{_RULE}\nPRETRAIN DATA  ({out_dir})\n{_RULE}")
    _print_manifest(out_dir)
    for i, r in enumerate(_preview(out_dir, n), 1):
        print(f"\n----- sample {i}  (numbers encoded: {r.get('num_numbers')}) -----")
        print("RAW    :", r["raw"][:600].replace("\n", " "))
        print("ENCODED:", r["encoded"][:900].replace("\n", " "))
        print("DECODED:", codec.decode_text(r["encoded"])[:600].replace("\n", " "))


def inspect_finetune(cfg: Config, n: int):
    out_dir = cfg.paths.finetune_data
    codec = NumberCodec.from_config(cfg)
    print(f"{_RULE}\nFINETUNE DATA  ({out_dir})\n{_RULE}")
    _print_manifest(out_dir)
    for i, r in enumerate(_preview(out_dir, n), 1):
        print(f"\n----- sample {i} -----")
        print("PROBLEM          :", r["problem"][:500].replace("\n", " "))
        print("ENCODED PROMPT   :", r["encoded_prompt"][:900].replace("\n", " "))
        print("ENCODED COMPLETION:", r["encoded_completion"][:900].replace("\n", " "))
        print("DECODED COMPLETION:", codec.decode_text(r["encoded_completion"])[:500].replace("\n", " "))


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect preprocessed data samples")
    p.add_argument("--split", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    if args.split == "pretrain":
        inspect_pretrain(cfg, args.n)
    else:
        inspect_finetune(cfg, args.n)


if __name__ == "__main__":
    main()
