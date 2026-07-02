"""Model loading, LoRA attach/merge, and batched generation."""

from __future__ import annotations

import os
from typing import List, Tuple

from numllm.utils import resolve_dtype


# ------------------------------------------------------------- base loading

def _quant_config(cfg):
    if not cfg.model.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(cfg, model_path=None, tokenizer=None, for_training=False):
    from transformers import AutoModelForCausalLM

    path = model_path or cfg.model.base_model
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=resolve_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
        quantization_config=_quant_config(cfg),
        trust_remote_code=cfg.model.trust_remote_code,
    )

    if tokenizer is not None:
        orig_vocab = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != orig_vocab:
            model.resize_token_embeddings(len(tokenizer))
            # Qwen pads its embedding table beyond the tokenizer, so a resize can
            # SHRINK it and place our new tokens on old (random) padding rows —
            # HF's mean-init only runs when growing. Initialize them explicitly.
            if cfg.encoding.enabled:
                init_new_token_embeddings(model, tokenizer, cfg)

    if for_training:
        if cfg.model.load_in_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True)
        model.config.use_cache = False
    return model


def init_new_token_embeddings(model, tokenizer, cfg):
    """Set the numeric special-token embedding rows to the mean of the other
    (pretrained) rows, so new tokens start neutral. They are then trained in
    full via lora.cpt_modules_to_save. Handles tied and untied lm_head, and is
    correct whether the resize grew or shrank the table."""
    import torch
    from numllm.tokenizer_utils import build_special_tokens

    ids = [i for i in tokenizer.convert_tokens_to_ids(build_special_tokens(cfg))
           if isinstance(i, int) and i >= 0]
    if not ids:
        return
    inp = model.get_input_embeddings().weight
    keep = torch.ones(inp.shape[0], dtype=torch.bool)
    keep[ids] = False
    with torch.no_grad():
        inp.data[ids] = inp.data[keep].float().mean(dim=0).to(inp.dtype)
        out = model.get_output_embeddings()
        if out is not None and out.weight.data_ptr() != inp.data_ptr():   # untied
            out.weight.data[ids] = out.weight.data[keep].float().mean(dim=0).to(out.weight.dtype)


def attach_lora(model, cfg, modules_to_save):
    import dataclasses
    from peft import LoraConfig, get_peft_model

    ms = list(modules_to_save) or None
    kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        modules_to_save=ms,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Qwen (and Llama-3.2-3B) tie embed_tokens<->lm_head. If we train them in
    # full, keep the tie enforced so the merged model doesn't drop the trained
    # lm_head on reload (PEFT #2777). Guarded for PEFT versions without the flag.
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    if tied and ms and ("embed_tokens" in ms or "lm_head" in ms):
        if any(f.name == "ensure_weight_tying" for f in dataclasses.fields(LoraConfig)):
            kwargs["ensure_weight_tying"] = True

    peft_model = get_peft_model(model, LoraConfig(**kwargs))
    peft_model.enable_input_require_grads()   # needed for gradient checkpointing
    peft_model.print_trainable_parameters()
    return peft_model


# ------------------------------------------------------------- merge

def merge_adapter(cfg, adapter_dir: str, out_dir: str, base_path=None,
                  tokenizer_dir=None):
    """Merge a LoRA adapter (incl. trained embed/lm_head) into a full model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = base_path or cfg.model.base_model
    tokenizer_dir = tokenizer_dir or (cfg.paths.tokenizer if cfg.encoding.enabled else base_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=resolve_dtype(cfg.model.dtype),
        trust_remote_code=cfg.model.trust_remote_code)
    if base.get_input_embeddings().weight.shape[0] != len(tok):
        base.resize_token_embeddings(len(tok))

    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    del merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


# ------------------------------------------------------------- resolve/load

def _is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def resolve_model_spec(cfg, spec: str) -> Tuple[str, bool]:
    """Map a friendly spec to (path, uses_number_encoding)."""
    if spec == "base":
        return cfg.model.base_model, False
    if spec == "cpt":
        merged = os.path.join(cfg.paths.cpt_out, "merged")
        return (merged if os.path.isdir(merged) else cfg.paths.cpt_out), cfg.encoding.enabled
    if spec == "sft":
        merged = os.path.join(cfg.paths.sft_out, "merged")
        return (merged if os.path.isdir(merged) else cfg.paths.sft_out), cfg.encoding.enabled
    return spec, cfg.encoding.enabled          # explicit path


def load_for_inference(cfg, spec: str):
    """Return (model, tokenizer, uses_encoding) ready for generate()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path, encoded = resolve_model_spec(cfg, spec)

    if _is_adapter_dir(path):
        from peft import PeftModel
        adapter_cfg_base = _read_adapter_base(path)
        tok_dir = path if _has_tokenizer(path) else cfg.paths.tokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir)
        base = AutoModelForCausalLM.from_pretrained(
            adapter_cfg_base, torch_dtype=resolve_dtype(cfg.model.dtype),
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code)
        if base.get_input_embeddings().weight.shape[0] != len(tok):
            base.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(base, path).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=resolve_dtype(cfg.model.dtype),
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tok, encoded


def _read_adapter_base(adapter_dir: str) -> str:
    import json
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)["base_model_name_or_path"]


def _has_tokenizer(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, f))
               for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"))


# ------------------------------------------------------------- generation

def generate(model, tok, prompts: List[str], max_new_tokens: int,
             temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
             max_prompt_tokens: int = 3072) -> List[str]:
    """Batched generation. Returns text generated *after* each prompt, with
    numeric special tokens preserved (core specials stripped)."""
    import torch

    prev_side, prev_trunc = tok.padding_side, tok.truncation_side
    tok.padding_side = "left"                  # correct for decoder-only batch gen
    tok.truncation_side = "left"               # keep the question end, drop old few-shot
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_prompt_tokens).to(model.device)

    do_sample = temperature and temperature > 0
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
        if top_k:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    tok.padding_side, tok.truncation_side = prev_side, prev_trunc
    gen = out[:, enc["input_ids"].shape[1]:]
    eos = tok.eos_token_id
    texts = []
    for row in gen.tolist():
        if eos in row:
            row = row[:row.index(eos)]
        texts.append(tok.decode(row, skip_special_tokens=False).strip())
    return texts
