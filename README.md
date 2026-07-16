# NumLLM

Continual pre-training + fine-tuning of **Qwen2.5** (**3B** or **1.5B** — pick
from the model registry) with a custom **numeric encoding** scheme. Numbers in
the training text are rewritten into structured special tokens (see
[`encode_decode.py`](encode_decode.py)) so the model learns digit/magnitude
structure explicitly.

Pipeline: **preprocess → continual pre-train (CPT) → evaluate → fine-tune (SFT)
→ infer → benchmark**. Every step is a separate command, uses LoRA, and
checkpoints so it can resume after a crash. Each model's weights are saved under
its own namespace, so you can train and benchmark several models side by side
(see [Choosing a model](#choosing-a-model-qwen25-3b--15b)).

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

Qwen2.5-3B and Qwen2.5-1.5B are openly downloadable (not gated). A
`huggingface-cli login` is only needed if you hit anonymous rate limits or use a
private cache:

```bash
huggingface-cli login   # optional
```

All outputs go under `artifacts/` (override with `paths.root`). Config lives in
[`configs/default.yaml`](configs/default.yaml); override any field on the CLI:

```bash
--set cpt.per_device_batch_size=4 --set encoding.type=bracket
```

---

## Choosing a model (Qwen2.5-3B / 1.5B)

The `models` registry in [`configs/default.yaml`](configs/default.yaml) lists the
base models you can run. `model.name` selects the active one and **namespaces
every per-model output**, so each model's tokenizer, continual-pretrained (CPT =
"pre-trained") and fine-tuned (SFT) weights, and eval results live in their own
folder and never overwrite each other:

```
artifacts/models/<name>/{tokenizer, cpt, sft}     # per-model weights
artifacts/eval/<name>/...                          # per-model benchmark results
```

The default is `qwen3b`. Switch the **entire** pipeline to the 1.5B model by
adding `--set model.name=qwen1.5b` to every command:

```bash
python -m numllm.train.continual_pretrain --set model.name=qwen1.5b
python -m numllm.train.finetune           --set model.name=qwen1.5b
python -m numllm.eval.run_benchmarks --model base,cpt,sft --set model.name=qwen1.5b
```

Preprocessed data (`artifacts/data/…`) is **shared** across models: it is encoded
*text*, independent of the tokenizer, so you preprocess FineWeb / NuminaMath once
and reuse it for every model. (The token counts that derive `cpt.max_steps` come
from whichever model preprocessed the data; Qwen2.5-3B and 1.5B share a tokenizer,
so they match exactly. For a model with a different tokenizer, set `cpt.max_steps`
explicitly.)

Add another model by giving it an id and any HF causal-LM base model:

```yaml
models:
  qwen3b:   { base_model: Qwen/Qwen2.5-3B }
  qwen1.5b: { base_model: Qwen/Qwen2.5-1.5B }
  mymodel:  { base_model: some-org/some-causal-lm }
```

The smaller 1.5B model has room for a bigger batch, e.g.
`--set cpt.per_device_batch_size=16 --set sft.per_device_batch_size=8`.

---

## The steps

### 1. Preprocess FineWeb (continual-pretrain data, ~1B tokens)

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

`--model` picks the **stage** (base | cpt | sft); `--set model.name=…` picks the
**base model**. They are independent axes, so you can benchmark every stage of
every model without anything overwriting anything else.

```bash
# one stage, all benchmarks (active model = qwen3b by default)
python -m numllm.eval.run_benchmarks --model sft

# compare base vs cpt vs sft on a quick subset
python -m numllm.eval.run_benchmarks --model base,cpt,sft \
    --benchmarks gsm8k,svamp,mmlu --set eval.limit=200

# the same sweep for the 1.5B model (results land in a separate folder)
python -m numllm.eval.run_benchmarks --model base,cpt,sft \
    --benchmarks gsm8k,svamp,mmlu --set eval.limit=200 --set model.name=qwen1.5b
```

Per-example records and per-benchmark summaries land in
`artifacts/eval/<model.name>/<stage>/`; a combined `all_results.json` per model
is written too.

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
artifacts/                    # (gitignored) all outputs
  data/{pretrain,finetune}/   #   shared across models (encoded text)
  models/<name>/tokenizer/    #   per-model extended tokenizer
  models/<name>/cpt/          #   per-model continual-pretrained weights (+ merged/)
  models/<name>/sft/          #   per-model fine-tuned weights (+ merged/)
  eval/<name>/                #   per-model benchmark results
  logs/<name>/
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
