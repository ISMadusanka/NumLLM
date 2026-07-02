"""Held-out perplexity / loss for a model variant, on encoded FineWeb text.

Useful to check that continual pre-training actually lowered LM loss on the
encoded distribution.

    python -m numllm.eval.perplexity --model cpt
    python -m numllm.eval.perplexity --model base --no-encoded

Note: encoded vs plain models tokenize differently, so per-token perplexity is
only directly comparable within the same encoding. `total_tokens` is reported
so bits-per-byte can be derived if a strict comparison is needed.
"""

from __future__ import annotations

import argparse
import math
import os

from numllm.config import Config
from numllm.encoding_utils import NumberCodec
from numllm.models.model_utils import load_for_inference
from numllm.utils import hard_exit, human, load_json, save_json, setup_logging


def _default_skip(cfg) -> int:
    manifest = os.path.join(cfg.paths.pretrain_data, "manifest.json")
    if os.path.exists(manifest):
        # start a little past the training data to reduce overlap
        return int(load_json(manifest).get("docs_seen", 0))
    return 200_000


def run(cfg, spec, num_docs, skip_docs, block_size, encoded_override, logger):
    import torch
    from datasets import load_dataset

    model, tok, encoded = load_for_inference(cfg, spec)
    if encoded_override is not None:
        encoded = encoded_override
    codec = NumberCodec.from_config(cfg)
    pd = cfg.pretrain_data

    if pd.name:
        ds = load_dataset(pd.dataset, pd.name, split=pd.split, streaming=True)
    else:
        ds = load_dataset(pd.dataset, split=pd.split, streaming=True)
    if skip_docs:
        logger.info(f"Skipping {human(skip_docs)} docs to reach held-out region ...")
        ds = ds.skip(skip_docs)

    buf, taken = [], 0
    for ex in ds:
        text = ex.get(pd.text_field) or ""
        if encoded:
            enc, cnt = codec.encode_text(text)
            if cnt < 1:
                continue
        else:
            enc = text
            if not text.strip():
                continue
        ids = tok(enc, add_special_tokens=False)["input_ids"]
        ids.append(tok.eos_token_id)
        buf.extend(ids)
        taken += 1
        if taken >= num_docs:
            break
    logger.info(f"Collected {human(len(buf))} tokens from {taken} held-out docs")

    total_nll, total_tok = 0.0, 0
    device = model.device
    model.eval()
    with torch.no_grad():
        for i in range(0, len(buf) - 1, block_size):
            chunk = buf[i:i + block_size + 1]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor([chunk[:-1]], device=device)
            labels = torch.tensor([chunk[1:]], device=device)
            logits = model(input_ids=input_ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
            total_nll += loss.item()
            total_tok += labels.numel()

    avg = total_nll / max(1, total_tok)
    result = {"model": spec, "encoded": encoded, "avg_loss": avg,
              "perplexity": math.exp(avg) if avg < 20 else float("inf"),
              "total_tokens": total_tok, "num_docs": taken}
    out = os.path.join(cfg.paths.eval_out, "perplexity", f"{spec.replace('/', '_')}.json")
    save_json(result, out)
    logger.info(f"[{spec}] avg_loss={avg:.4f} ppl={result['perplexity']:.3f} "
                f"tokens={human(total_tok)} -> {out}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Held-out perplexity evaluation")
    p.add_argument("--model", default="cpt", help="base | cpt | sft | <path>")
    p.add_argument("--num-docs", type=int, default=2000)
    p.add_argument("--skip-docs", type=int, default=None,
                   help="docs to skip first (default: past the training region)")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--encoded", dest="encoded", action="store_true", default=None)
    p.add_argument("--no-encoded", dest="encoded", action="store_false")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    logger = setup_logging("perplexity", cfg.paths.logs)
    skip = args.skip_docs if args.skip_docs is not None else _default_skip(cfg)
    block = args.block_size or cfg.cpt.block_size
    run(cfg, args.model, args.num_docs, skip, block, args.encoded, logger)
    hard_exit(0)     # avoid streaming thread crash during shutdown


if __name__ == "__main__":
    main()
