"""Typed, YAML-backed configuration for the whole pipeline.

Every step loads a single Config. Fields have sensible defaults, so the
YAML only needs to override what you care about. Dotted CLI overrides are
supported too, e.g.  --set encoding.type=bracket --set cpt.lr=1e-4
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Optional

import yaml


# ---------------------------------------------------------------- sections

@dataclass
class ModelCfg:
    base_model: str = "Qwen/Qwen2.5-3B"
    dtype: str = "bfloat16"                 # bfloat16 | float16 | float32
    attn_implementation: str = "sdpa"       # sdpa | flash_attention_2 | eager
    load_in_4bit: bool = False              # QLoRA-style base loading
    trust_remote_code: bool = False


@dataclass
class EncodingCfg:
    type: str = "flat"                      # flat | tree | bracket | hierarchy
    enabled: bool = True                    # master on/off for number encoding
    max_group_index: int = 40               # special-token & int-group coverage
    max_frac_groups: int = 20
    max_int_digits: int = 30                # skip "numbers" longer than this
    max_frac_digits: int = 15


@dataclass
class PathsCfg:
    root: str = "artifacts"
    pretrain_data: Optional[str] = None
    finetune_data: Optional[str] = None
    tokenizer: Optional[str] = None
    cpt_out: Optional[str] = None
    sft_out: Optional[str] = None
    eval_out: Optional[str] = None
    logs: Optional[str] = None

    def resolve(self) -> "PathsCfg":
        r = self.root
        self.pretrain_data = self.pretrain_data or os.path.join(r, "data", "pretrain")
        self.finetune_data = self.finetune_data or os.path.join(r, "data", "finetune")
        self.tokenizer = self.tokenizer or os.path.join(r, "tokenizer")
        self.cpt_out = self.cpt_out or os.path.join(r, "models", "cpt")
        self.sft_out = self.sft_out or os.path.join(r, "models", "sft")
        self.eval_out = self.eval_out or os.path.join(r, "eval")
        self.logs = self.logs or os.path.join(r, "logs")
        return self


@dataclass
class PretrainDataCfg:
    dataset: str = "HuggingFaceFW/fineweb"
    name: Optional[str] = "sample-10BT"     # a ~10BT slice; we take ~5B from it
    split: str = "train"
    text_field: str = "text"
    target_tokens: int = 5_000_000_000      # stop preprocessing here
    docs_per_shard: int = 50_000            # JSONL shard size (accepted docs)
    preview_samples: int = 50               # raw->encoded pairs saved for inspection
    min_numbers_per_doc: int = 1            # skip docs with fewer numbers than this
    state_every_docs: int = 20_000          # checkpoint stream progress this often
    seed: int = 1234


@dataclass
class FinetuneDataCfg:
    dataset: str = "AI-MO/NuminaMath-CoT"
    name: Optional[str] = None
    split: str = "train"
    problem_field: str = "problem"
    solution_field: str = "solution"
    target_tokens: int = 10_000_000
    docs_per_shard: int = 20_000
    preview_samples: int = 50
    # {problem} is filled in; the model is trained to produce the solution.
    prompt_template: str = (
        "Solve the following math problem. Show your reasoning and give the "
        "final answer.\n\nProblem:\n{problem}\n\nSolution:\n"
    )
    encode_solution: bool = True            # also encode numbers in the target
    state_every_docs: int = 10_000
    seed: int = 1234


@dataclass
class LoraCfg:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # trained-in-full (not low-rank). Needed for the new numeric tokens' rows.
    cpt_modules_to_save: List[str] = field(default_factory=lambda: [
        "embed_tokens", "lm_head",
    ])
    sft_modules_to_save: List[str] = field(default_factory=list)


@dataclass
class CptCfg:
    block_size: int = 1024
    per_device_batch_size: int = 8
    grad_accum: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.02
    max_steps: int = -1                     # -1 => derive from target_tokens
    num_train_epochs: float = 1.0           # only used if max_steps < 0 and no token target
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 0                     # 0 disables mid-training eval
    gradient_checkpointing: bool = True
    merge_after: bool = True                # write a merged full model for downstream use
    seed: int = 1234


@dataclass
class SftCfg:
    max_seq_len: int = 2048
    per_device_batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    # which model to fine-tune on top of: "cpt" (merged CPT model) or "base"
    start_from: str = "cpt"
    merge_after: bool = True
    seed: int = 1234


@dataclass
class EvalCfg:
    benchmarks: List[str] = field(default_factory=lambda: [
        "gsm8k", "svamp", "math", "mmlu_pro_math",
        "drop", "tatqa", "finqa", "mmlu",
    ])
    limit: int = 0                          # 0 => full split, else first N examples
    n_shots: int = 4
    max_new_tokens: int = 512
    batch_size: int = 8
    temperature: float = 0.0                # 0 => greedy
    top_p: float = 1.0
    # Override the HF dataset id used for a benchmark, e.g.
    #   dataset_overrides: {tatqa: "some-org/tat-qa"}
    dataset_overrides: dict = field(default_factory=dict)
    seed: int = 1234


@dataclass
class InferCfg:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


# ---------------------------------------------------------------- root

@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    encoding: EncodingCfg = field(default_factory=EncodingCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    pretrain_data: PretrainDataCfg = field(default_factory=PretrainDataCfg)
    finetune_data: FinetuneDataCfg = field(default_factory=FinetuneDataCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    cpt: CptCfg = field(default_factory=CptCfg)
    sft: SftCfg = field(default_factory=SftCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    infer: InferCfg = field(default_factory=InferCfg)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[str] = None,
             overrides: Optional[List[str]] = None) -> "Config":
        cfg = cls()
        data = {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        for sec in fields(cls):
            section_obj = getattr(cfg, sec.name)
            if is_dataclass(section_obj) and isinstance(data.get(sec.name), dict):
                _apply_dict(section_obj, data[sec.name], sec.name)
        for other in set(data) - {f.name for f in fields(cls)}:
            warnings.warn(f"Unknown config section ignored: {other!r}")
        if overrides:
            for ov in overrides:
                cfg.apply_override(ov)
        cfg.paths.resolve()
        return cfg

    def apply_override(self, dotted: str) -> None:
        """Apply 'section.key=value' (value parsed as YAML scalar/list)."""
        if "=" not in dotted:
            raise ValueError(f"Override must be section.key=value, got {dotted!r}")
        key, raw = dotted.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.key, got {key!r}")
        section, name = parts
        section_obj = getattr(self, section, None)
        if not is_dataclass(section_obj):
            raise ValueError(f"Unknown config section: {section!r}")
        valid = {f.name for f in fields(section_obj)}
        if name not in valid:
            raise ValueError(f"Unknown key {name!r} in section {section!r}")
        setattr(section_obj, name, yaml.safe_load(raw))

    def to_dict(self) -> dict:
        return {f.name: _section_to_dict(getattr(self, f.name)) for f in fields(self)}


def _apply_dict(section_obj, values: dict, section_name: str) -> None:
    valid = {f.name for f in fields(section_obj)}
    for k, v in values.items():
        if k in valid:
            setattr(section_obj, k, v)
        else:
            warnings.warn(f"Unknown key {k!r} in config section {section_name!r} ignored")


def _section_to_dict(section_obj):
    if is_dataclass(section_obj):
        return {f.name: getattr(section_obj, f.name) for f in fields(section_obj)}
    return section_obj
