"""Stream FineWeb, keep docs that contain numbers, encode those numbers,
and write inspectable JSONL shards until ~target_tokens is reached.

Streaming means we never download the whole dataset — shards are pulled and
processed on the fly, and preprocessing is crash-resumable via state.json.

Run:
    python -m numllm.data.preprocess_pretrain --config configs/default.yaml
Resume: just run the same command again.
"""

from __future__ import annotations

import argparse
import os

from numllm.config import Config
from numllm.encoding_utils import NumberCodec
from numllm.tokenizer_utils import get_extended_tokenizer
from numllm.utils import (JsonlShardWriter, StateStore, hard_exit, human,
                          save_json, set_seed, setup_logging)

_TOK_BATCH = 1000          # docs tokenized per batch (for token counting)


def _load_stream(dataset: str, name, split: str):
    from datasets import load_dataset
    if name:
        return load_dataset(dataset, name, split=split, streaming=True)
    return load_dataset(dataset, split=split, streaming=True)


def _resume_stream(ds, state: StateStore):
    hf_state = state.get("hf_state")
    if hf_state is not None and hasattr(ds, "load_state_dict"):
        try:
            ds.load_state_dict(hf_state)
            return ds, "state_dict"
        except Exception:
            pass
    docs_seen = state.get("docs_seen", 0)
    if docs_seen:
        return ds.skip(docs_seen), "skip"
    return ds, "fresh"


def run(cfg: Config, logger):
    set_seed(cfg.pretrain_data.seed)
    pd = cfg.pretrain_data
    out_dir = cfg.paths.pretrain_data
    os.makedirs(out_dir, exist_ok=True)

    tok, added = get_extended_tokenizer(cfg)
    logger.info(f"Tokenizer ready (added {added} numeric special tokens; vocab={len(tok)})")
    codec = NumberCodec.from_config(cfg)

    state = StateStore(os.path.join(out_dir, "state.json"))
    tokens_written = state.get("tokens_written", 0)
    docs_seen = state.get("docs_seen", 0)
    accepted = state.get("accepted_docs", 0)
    preview_written = state.get("preview_written", 0)
    target = pd.target_tokens

    if state.get("done"):
        logger.info(f"Already complete: {human(tokens_written)} tokens in {out_dir}")
        return

    ds = _load_stream(pd.dataset, pd.name, pd.split)
    ds, mode = _resume_stream(ds, state)
    logger.info(f"Streaming {pd.dataset}:{pd.name} split={pd.split} (resume={mode}, "
                f"docs_seen={docs_seen}, tokens={human(tokens_written)}/{human(target)})")

    writer = JsonlShardWriter(out_dir, "pretrain", pd.docs_per_shard,
                              start_shard=state.get("shard_idx", 0))
    writer.count_in_shard = state.get("count_in_shard", 0)
    preview_fh = open(os.path.join(out_dir, "preview.jsonl"), "a", encoding="utf-8")

    buf = []

    def flush() -> bool:
        """Tokenize buffered docs, write them, update counters. True if target hit."""
        nonlocal tokens_written, accepted
        if not buf:
            return False
        all_ids = tok(buf, add_special_tokens=False)["input_ids"]
        reached = False
        for text, ids in zip(buf, all_ids):
            writer.write({"text": text})
            tokens_written += len(ids) + 1        # +1 for the EOS added at train time
            accepted += 1
            if tokens_written >= target:
                reached = True
                break
        buf.clear()
        return reached

    def checkpoint():
        writer.flush()
        preview_fh.flush()
        hf_state = None
        if hasattr(ds, "state_dict"):
            try:
                import json as _json
                hf_state = ds.state_dict()
                _json.dumps(hf_state)          # ensure serializable
            except Exception:
                hf_state = None
        state.update(tokens_written=tokens_written, docs_seen=docs_seen,
                     accepted_docs=accepted, preview_written=preview_written,
                     shard_idx=writer.shard_idx, count_in_shard=writer.count_in_shard,
                     hf_state=hf_state, target_tokens=target,
                     done=tokens_written >= target)
        state.save()

    stop = False
    try:
        for ex in ds:
            docs_seen += 1
            text = ex.get(pd.text_field) or ""
            cnt = codec.count_numbers(text)          # regex count (works even if encoding off)
            if cnt < pd.min_numbers_per_doc:
                if docs_seen % pd.state_every_docs == 0:
                    checkpoint()
                    logger.info(f"seen={human(docs_seen)} accepted={human(accepted)} "
                                f"tokens={human(tokens_written)}/{human(target)}")
                continue
            encoded = codec.encode_text(text)[0]     # no-op when encoding disabled

            if preview_written < pd.preview_samples:
                import json as _json
                preview_fh.write(_json.dumps(
                    {"raw": text[:2000], "encoded": encoded[:4000], "num_numbers": cnt},
                    ensure_ascii=False) + "\n")
                preview_written += 1

            buf.append(encoded)
            if len(buf) >= _TOK_BATCH:
                stop = flush()
            if docs_seen % pd.state_every_docs == 0:
                if buf:
                    stop = flush() or stop
                checkpoint()
                logger.info(f"seen={human(docs_seen)} accepted={human(accepted)} "
                            f"tokens={human(tokens_written)}/{human(target)}")
            if stop:
                break
    finally:
        flush()
        checkpoint()
        writer.close()
        preview_fh.close()

    save_json({
        "dataset": f"{pd.dataset}:{pd.name}",
        "split": pd.split,
        "encoding": cfg.encoding.type,
        "target_tokens": target,
        "tokens_written": tokens_written,
        "docs_seen": docs_seen,
        "accepted_docs": accepted,
        "num_shards": writer.shard_idx + 1,
        "done": tokens_written >= target,
    }, os.path.join(out_dir, "manifest.json"))
    logger.info(f"DONE: {human(tokens_written)} tokens, {human(accepted)} docs "
                f"(from {human(docs_seen)} seen) -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Preprocess FineWeb for continual pretraining")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], help="dotted override key=value")
    args = p.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    logger = setup_logging("preprocess_pretrain", cfg.paths.logs)
    run(cfg, logger)
    hard_exit(0)     # avoid streaming/tokenizer thread crash during shutdown


if __name__ == "__main__":
    main()
