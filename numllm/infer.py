"""Run inference against any model variant.

    python -m numllm.infer --model sft --prompt "What is 12345 + 6789?"
    python -m numllm.infer --model cpt --interactive
    python -m numllm.infer --model base --prompt "..."     # plain numbers

For encoded models the prompt's numbers are encoded before generation and the
output's numbers are decoded back to plain digits.
"""

from __future__ import annotations

import argparse

from numllm.config import Config
from numllm.encoding_utils import NumberCodec
from numllm.models.model_utils import generate, load_for_inference
from numllm.utils import setup_logging


def format_prompt(cfg: Config, spec: str, prompt: str, use_template) -> str:
    want = use_template if use_template is not None else (spec == "sft")
    if want:
        return cfg.finetune_data.prompt_template.format(problem=prompt)
    return prompt


def run_once(cfg, model, tok, codec, encoded, spec, prompt, use_template):
    text = format_prompt(cfg, spec, prompt, use_template)
    model_input = codec.encode_text(text)[0] if encoded else text
    out = generate(model, tok, [model_input],
                   max_new_tokens=cfg.infer.max_new_tokens,
                   temperature=cfg.infer.temperature,
                   top_p=cfg.infer.top_p, top_k=cfg.infer.top_k)[0]
    decoded = codec.decode_text(out) if encoded else out
    return out, decoded


def main(argv=None):
    p = argparse.ArgumentParser(description="Inference")
    p.add_argument("--model", default="sft", help="base | cpt | sft | <path>")
    p.add_argument("--prompt", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--template", dest="template", action="store_true", default=None,
                   help="force the SFT prompt template")
    p.add_argument("--no-template", dest="template", action="store_false")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    logger = setup_logging("infer", cfg.paths.logs)
    logger.info(f"Loading model spec={args.model} ...")
    model, tok, encoded = load_for_inference(cfg, args.model)
    codec = NumberCodec.from_config(cfg)
    logger.info(f"Ready (number-encoding {'ON' if encoded else 'OFF'}).")

    if args.interactive:
        print("Enter a prompt (blank line or Ctrl-D to quit):")
        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            raw, decoded = run_once(cfg, model, tok, codec, encoded, args.model,
                                    prompt, args.template)
            if encoded:
                print("\n[encoded output]\n" + raw)
            print("\n[answer]\n" + decoded + "\n")
    else:
        if not args.prompt:
            p.error("provide --prompt or --interactive")
        raw, decoded = run_once(cfg, model, tok, codec, encoded, args.model,
                                args.prompt, args.template)
        if encoded:
            print("\n[encoded output]\n" + raw)
        print("\n[answer]\n" + decoded)


if __name__ == "__main__":
    main()
