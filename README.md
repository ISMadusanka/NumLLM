# NumLLM

Continual pre-training + fine-tuning of **Llama-3.2-3B** (configurable) with a
custom **numeric encoding** scheme. Numbers in the training text are rewritten
into structured special tokens (see [`encode_decode.py`](encode_decode.py)) so
the model learns digit/magnitude structure explicitly.

Pipeline: **preprocess → continual pre-train (CPT) → evaluate → fine-tune (SFT)
→ infer → benchmark**. Every step is a separate command, uses LoRA, and
checkpoints so it can resume after a crash.

---

## How the numeric encoding works

`encode_decode.py` turns one number into tokens. Example (FLAT):

```
1000000   ->  <NUM><M>1</M><K>000</K><U>000</U></NUM>
-42.5     ->  <NEG><NUM><U>42</U><FRAC><F1>5</F1></FRAC></NUM></NEG>
```

The magnitude tags (`<U> <K> <M> <B> …`) come from `HIERARCHY_NAMES`, and
`generate_special_tokens()` produces the full vocabulary that gets **added to
the tokenizer** as new tokens. Because those token embeddings start untrained,
CPT trains the embedding + LM head in full (`lora.cpt_modules_to_save`).

`numllm/encoding_utils.py` applies this to *free text*: it finds real numbers
(skipping versions, IPs, ids/hashes) and swaps them for their encoding, and can
decode a model's output back to plain digits.

Encoding is configurable: `encoding.type` = `flat` (default) | `tree` |
`bracket` | `hierarchy`, and `encoding.enabled: false` trains on plain numbers.

---

## Setup

RTX 5090 is Blackwell (sm_120) — install a CUDA 12.8 PyTorch build **first**:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt
```

Llama is gated on Hugging Face — log in and accept the license once:

```bash
huggingface-cli login
```

All outputs go under `artifacts/` (override with `paths.root`). Config lives in
[`configs/default.yaml`](configs/default.yaml); override any field on the CLI:

```bash
--set cpt.per_device_batch_size=4 --set encoding.type=bracket
```

---

## The steps

### 1. Preprocess FineWeb (continual-pretrain data, ~5B tokens)

Streams FineWeb (never downloads it whole), keeps docs containing numbers,
encodes the numbers, and stops at `pretrain_data.target_tokens`. Resumable.

```bash
python -m numllm.data.preprocess_pretrain --config configs/default.yaml
# smaller smoke test:
python -m numllm.data.preprocess_pretrain --set pretrain_data.target_tokens=50000000
```

Inspect samples (raw → encoded, with round-trip decode):

```bash
python -m numllm.data.inspect_data --split pretrain --n 5
```

### 2. Preprocess NuminaMath (fine-tune data, ~10M tokens)

```bash
python -m numllm.data.preprocess_finetune --config configs/default.yaml
python -m numllm.data.inspect_data --split finetune --n 5
```

### 3. Continual pre-training (LoRA)

Trains on packed encoded FineWeb. `max_steps` is derived from the token target
unless you set `cpt.max_steps`. Logs loss/lr each `logging_steps`; checkpoints
every `save_steps`. Re-run the same command to resume.

```bash
python -m numllm.train.continual_pretrain --config configs/default.yaml
```

Produces a LoRA adapter in `artifacts/models/cpt/` and (if `cpt.merge_after`)
a merged full model in `artifacts/models/cpt/merged/`.

### 4. Evaluate the CPT model (held-out perplexity)

```bash
python -m numllm.eval.perplexity --model cpt
python -m numllm.eval.perplexity --model base --no-encoded   # baseline
```

### 5. Fine-tune on NuminaMath (LoRA, on top of the merged CPT model)

Loss is masked to completion tokens only. Resumable.

```bash
python -m numllm.train.finetune --config configs/default.yaml
# to fine-tune the raw base model instead of the CPT one:
python -m numllm.train.finetune --set sft.start_from=base
```

Produces `artifacts/models/sft/` (+ `merged/`).

### 6. Inference

```bash
python -m numllm.infer --model sft --prompt "A shop sold 1240 items at 3.5 each. Revenue?"
python -m numllm.infer --model cpt --interactive
python -m numllm.infer --model base --prompt "..."      # plain numbers, no encoding
```

Input numbers are encoded before generation; output numbers are decoded back.

### 7. Benchmarks

GSM8K, SVAMP, MATH, MMLU-Pro (math), DROP, TAT-QA, FinQA, MMLU. Base model is
scored on plain numbers; CPT/SFT on encoded numbers.

```bash
# one model, all benchmarks
python -m numllm.eval.run_benchmarks --model sft

# compare base vs cpt vs sft on a quick subset
python -m numllm.eval.run_benchmarks --model base,cpt,sft \
    --benchmarks gsm8k,svamp,mmlu --set eval.limit=200
```

Per-example records and per-benchmark summaries land in
`artifacts/eval/<model>/`; a combined `all_results.json` is written too.

**TAT-QA** has no canonical HF id and is skipped unless you provide one:

```yaml
eval:
  dataset_overrides: {tatqa: "your-org/tat-qa"}
```

A benchmark whose dataset fails to load is logged and skipped (the run
continues).

---

## Folder layout

```
encode_decode.py              # numeric encode/decode (core scheme)
configs/default.yaml          # every knob
numllm/
  config.py utils.py encoding_utils.py tokenizer_utils.py
  data/    preprocess_pretrain.py preprocess_finetune.py
           lm_dataset.py sft_dataset.py inspect_data.py
  models/  model_utils.py
  train/   continual_pretrain.py finetune.py common.py
  eval/    perplexity.py run_benchmarks.py benchmarks/*.py
  infer.py
artifacts/                    # (gitignored) data, checkpoints, models, eval, logs
```

---

## Resuming & checkpoints

- **Preprocessing** writes `state.json` (stream position, token count, shard)
  every `state_every_docs`. Re-running continues where it stopped. Up to one
  checkpoint interval of documents may be reprocessed after a hard crash
  (harmless duplication).
- **Training** uses HF `Trainer` checkpoints (`save_steps`, `save_total_limit`).
  Re-running the same command resumes from the last checkpoint automatically.

## Memory tuning (single 32 GB GPU)

Defaults target a 32 GB card with bf16 + gradient checkpointing + LoRA. If you
hit OOM in CPT (training the full embedding/LM-head for the new tokens is the
heavy part):

- lower `cpt.per_device_batch_size` / raise `cpt.grad_accum`,
- lower `cpt.block_size`,
- set `model.load_in_4bit: true` (QLoRA),
- or trim `lora.cpt_modules_to_save` (note: the new numeric tokens won't learn
  well if their embeddings aren't trained).
```
