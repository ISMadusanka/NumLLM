"""Run downstream benchmarks for one or more model variants.

    # continual-pretrained model, all benchmarks from the config
    python -m numllm.eval.run_benchmarks --model cpt

    # compare base / cpt / sft on two benchmarks, 200 examples each
    python -m numllm.eval.run_benchmarks --model base,cpt,sft \
        --benchmarks gsm8k,svamp --set eval.limit=200

The base model is evaluated on plain numbers; cpt/sft on encoded numbers
(override with --encoded / --no-encoded).
"""

from __future__ import annotations

import argparse
import os

from numllm.config import Config
from numllm.encoding_utils import NumberCodec
from numllm.models.model_utils import generate, load_for_inference
from numllm.utils import save_json, setup_logging

from numllm.eval.benchmarks.gsm8k import GSM8K
from numllm.eval.benchmarks.svamp import SVAMP
from numllm.eval.benchmarks.math_bench import MATH
from numllm.eval.benchmarks.mmlu import MMLU
from numllm.eval.benchmarks.mmlu_pro_math import MMLUProMath
from numllm.eval.benchmarks.drop import DROP
from numllm.eval.benchmarks.tatqa import TATQA
from numllm.eval.benchmarks.finqa import FinQA

REGISTRY = {c.name: c for c in
            [GSM8K, SVAMP, MATH, MMLUProMath, DROP, TATQA, FinQA, MMLU]}


def run_model(cfg, spec, bench_names, encoded_override, logger):
    import torch

    logger.info(f"===== loading model '{spec}' =====")
    model, tok, encoded = load_for_inference(cfg, spec)
    if encoded_override is not None:
        encoded = encoded_override
    codec = NumberCodec.from_config(cfg)
    logger.info(f"model ready; number-encoding {'ON' if encoded else 'OFF'}")

    def gen_fn(prompts, max_new_tokens, temperature, top_p):
        return generate(model, tok, prompts, max_new_tokens, temperature, top_p)

    out_root = os.path.join(cfg.paths.eval_out, spec.replace("/", "_"))
    results = {}
    for name in bench_names:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning(f"unknown benchmark '{name}' (known: {list(REGISTRY)})")
            continue
        bench = cls(cfg, codec, encoded)
        try:
            bench.load()
        except Exception as ex:                     # dataset missing/id wrong
            logger.warning(f"[{name}] SKIPPED — load failed: {ex}")
            continue
        logger.info(f"[{name}] {len(bench.eval_examples)} eval examples")
        try:
            results[name] = bench.evaluate(
                gen_fn, logger, out_root,
                limit=cfg.eval.limit, n_shots=cfg.eval.n_shots,
                max_new_tokens=cfg.eval.max_new_tokens,
                temperature=cfg.eval.temperature, top_p=cfg.eval.top_p,
                batch_size=cfg.eval.batch_size)
        except Exception as ex:
            logger.warning(f"[{name}] SKIPPED — eval failed: {ex}")

    save_json({"model": spec, "encoded": encoded, "results": results},
              os.path.join(out_root, "results.json"))
    logger.info(f"===== RESULTS ({spec}) =====")
    for name, s in results.items():
        line = f"  {name:16s} acc={s['accuracy']:.4f}  (n={s['n']})"
        if "f1" in s:
            line += f"  f1={s['f1']:.4f}"
        logger.info(line)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Run downstream benchmarks")
    p.add_argument("--model", default="base", help="base|cpt|sft|<path> (comma-separated for several)")
    p.add_argument("--benchmarks", default=None, help="comma list; default = config")
    p.add_argument("--encoded", dest="encoded", action="store_true", default=None)
    p.add_argument("--no-encoded", dest="encoded", action="store_false")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    logger = setup_logging("eval_benchmarks", cfg.paths.logs)
    bench_names = ([b.strip() for b in args.benchmarks.split(",")]
                   if args.benchmarks else cfg.eval.benchmarks)
    all_results = {}
    for spec in args.model.split(","):
        all_results[spec.strip()] = run_model(cfg, spec.strip(), bench_names,
                                               args.encoded, logger)
    save_json(all_results, os.path.join(cfg.paths.eval_out, "all_results.json"))


if __name__ == "__main__":
    main()
