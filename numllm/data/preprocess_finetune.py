"""Stream NuminaMath, format each row as (prompt, completion), encode the
numbers, and write JSONL shards until ~target_tokens is reached.

Same streaming + crash-resume behaviour as the pretrain preprocessor.

Run:
    python -m numllm.data.preprocess_finetune --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os

from numllm.config import Config
from numllm.encoding_utils import NumberCodec
from numllm.tokenizer_utils import get_extended_tokenizer
from numllm.utils import (JsonlShardWriter, StateStore, human, save_json,
                          set_seed, setup_logging)

_TOK_BATCH = 500


def _load_stream(dataset, name, split):
    from datasets import load_dataset
    if name:
        return load_dataset(dataset, name, split=split, streaming=True)
    return load_dataset(dataset, split=split, streaming=True)


def _resume_stream(ds, state):
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
    set_seed(cfg.finetune_data.seed)
    fd = cfg.finetune_data
    out_dir = cfg.paths.finetune_data
    os.makedirs(out_dir, exist_ok=True)

    tok, added = get_extended_tokenizer(cfg)
    logger.info(f"Tokenizer ready (added {added} numeric special tokens; vocab={len(tok)})")
    codec = NumberCodec.from_config(cfg)

    state = StateStore(os.path.join(out_dir, "state.json"))
    tokens_written = state.get("tokens_written", 0)
    docs_seen = state.get("docs_seen", 0)
    accepted = state.get("accepted_docs", 0)
    preview_written = state.get("preview_written", 0)
    target = fd.target_tokens

    if state.get("done"):
        logger.info(f"Already complete: {human(tokens_written)} tokens in {out_dir}")
        return

    ds = _load_stream(fd.dataset, fd.name, fd.split)
    ds, mode = _resume_stream(ds, state)
    logger.info(f"Streaming {fd.dataset} split={fd.split} (resume={mode}, "
                f"tokens={human(tokens_written)}/{human(target)})")

    writer = JsonlShardWriter(out_dir, "finetune", fd.docs_per_shard,
                              start_shard=state.get("shard_idx", 0))
    writer.count_in_shard = state.get("count_in_shard", 0)
    preview_fh = open(os.path.join(out_dir, "preview.jsonl"), "a", encoding="utf-8")

    buf = []          # list of (prompt, completion)

    def flush() -> bool:
        nonlocal tokens_written, accepted
        if not buf:
            return False
        prompts = [p for p, _ in buf]
        comps = [c for _, c in buf]
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids = tok(comps, add_special_tokens=False)["input_ids"]
        reached = False
        for (p, c), pi, ci in zip(buf, p_ids, c_ids):
            writer.write({"prompt": p, "completion": c})
            tokens_written += len(pi) + len(ci) + 1     # +1 EOS
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
                hf_state = ds.state_dict()
                json.dumps(hf_state)
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
            problem = (ex.get(fd.problem_field) or "").strip()
            solution = (ex.get(fd.solution_field) or "").strip()
            if not problem or not solution:
                continue

            if (codec.count_numbers(problem) + codec.count_numbers(solution)) < 1:
                continue                   # keep only rows that actually contain numbers
            enc_problem = codec.encode_text(problem)[0]           # no-op if encoding off
            enc_solution = codec.encode_text(solution)[0] if fd.encode_solution else solution

            prompt = fd.prompt_template.format(problem=enc_problem)
            completion = enc_solution

            if preview_written < fd.preview_samples:
                preview_fh.write(json.dumps({
                    "problem": problem[:1500], "solution": solution[:2500],
                    "encoded_prompt": prompt[:3000], "encoded_completion": completion[:3000],
                }, ensure_ascii=False) + "\n")
                preview_written += 1

            buf.append((prompt, completion))
            if len(buf) >= _TOK_BATCH:
                stop = flush()
            if docs_seen % fd.state_every_docs == 0:
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
        "dataset": fd.dataset,
        "split": fd.split,
        "encoding": cfg.encoding.type,
        "target_tokens": target,
        "tokens_written": tokens_written,
        "docs_seen": docs_seen,
        "accepted_docs": accepted,
        "num_shards": writer.shard_idx + 1,
        "done": tokens_written >= target,
    }, os.path.join(out_dir, "manifest.json"))
    logger.info(f"DONE: {human(tokens_written)} tokens, {human(accepted)} examples "
                f"-> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Preprocess NuminaMath for fine-tuning")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    logger = setup_logging("preprocess_finetune", cfg.paths.logs)
    run(cfg, logger)


if __name__ == "__main__":
    main()
