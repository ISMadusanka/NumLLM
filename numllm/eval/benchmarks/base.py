"""Base class for the benchmark harness.

Protocol shared by every benchmark:
  * The model is instructed to finish with a line: "The answer is <ANSWER>."
  * Few-shot exemplars demonstrate that format (with dataset CoT where useful).
  * For encoded models the whole prompt is number-encoded before generation and
    the generation is number-decoded before the answer is extracted/scored.

A subclass sets the dataset ids/splits and implements `normalize(raw_example)`
returning a dict with:
    number/text : {"question": str, "answer": str, "golds": [str,...], "cot": str?}
    mc          : {"question": str, "choices": [str,...], "answer": "A", "cot": str?}
Return None to drop an example (e.g. wrong category).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from numllm.utils import save_json

QA_TEMPLATE = "Question:\n{q}\nAnswer:\n{a}"
INSTRUCTION_NUMERIC = ("Solve the problem step by step. On the last line write "
                       "exactly: The answer is <ANSWER>.")
INSTRUCTION_MC = ("Answer the multiple-choice question. On the last line write "
                  "exactly: The answer is <LETTER>.")


# ------------------------------------------------------------- text helpers

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def last_answer_span(text: str) -> str:
    m = list(re.finditer(r"answer\s+is\s*:?\s*(.+)", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip().strip(".").strip()
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def normalize_number(s: str) -> str:
    s = s.strip()
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        s = m.group(1)
    s = s.replace(",", "").replace("$", "").replace("%", "").replace("\\", "").strip()
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else s.strip()


def _to_float(x: str) -> float:
    x = x.strip()
    if "/" in x:
        n, d = x.split("/", 1)
        return float(n) / float(d)
    return float(x)


def numbers_equal(a: str, b: str, tol: float = 1e-3) -> bool:
    try:
        fa, fb = _to_float(a), _to_float(b)
    except (ValueError, ZeroDivisionError):
        return a.strip() == b.strip()
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def normalize_choice(s: str) -> str:
    m = re.search(r"\(?\b([A-J])\b\)?", s.strip())
    return m.group(1).upper() if m else s.strip().upper()


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------- base class

class Benchmark:
    name = "base"
    answer_type = "number"          # number | mc | text
    DEFAULT_DATASET = ""
    DATASET_CONFIG: Optional[str] = None
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT: Optional[str] = None      # None => split off the eval set
    FEWSHOT_POOL = 32

    def __init__(self, cfg, codec, encoded: bool):
        self.cfg = cfg
        self.codec = codec
        self.encoded = encoded
        self.dataset_id = cfg.eval.dataset_overrides.get(self.name, self.DEFAULT_DATASET)
        self.eval_examples: List[dict] = []
        self.fewshot_examples: List[dict] = []

    # ---- data
    def _load(self, split):
        from datasets import load_dataset
        pos = [self.DATASET_CONFIG] if self.DATASET_CONFIG else []
        return load_dataset(self.dataset_id, *pos, split=split)

    def normalize(self, raw) -> Optional[dict]:
        raise NotImplementedError

    def load(self):
        eval_raw = self._load(self.EVAL_SPLIT)
        ev = [self.normalize(e) for e in eval_raw]
        ev = [e for e in ev if e]
        if self.FEWSHOT_SPLIT:
            fs_raw = self._load(self.FEWSHOT_SPLIT)
            fs = [self.normalize(e) for e in fs_raw]
            self.fewshot_examples = [e for e in fs if e]
            self.eval_examples = ev
        else:
            self.fewshot_examples = ev[:self.FEWSHOT_POOL]
            self.eval_examples = ev[self.FEWSHOT_POOL:]
        return self

    # ---- prompt formatting
    def instruction(self) -> str:
        return INSTRUCTION_MC if self.answer_type == "mc" else INSTRUCTION_NUMERIC

    def gold_str(self, ex: dict) -> str:
        return f"({ex['answer']})" if self.answer_type == "mc" else str(ex["answer"])

    def _question_block(self, ex: dict) -> str:
        if self.answer_type == "mc":
            opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(ex["choices"]))
            return f"{ex['question']}\n{opts}"
        return ex["question"]

    def format_example(self, ex: dict, with_answer: bool) -> str:
        q = self._question_block(ex)
        if not with_answer:
            return QA_TEMPLATE.format(q=q, a="")
        a = ""
        if ex.get("cot"):
            a += ex["cot"].strip() + "\n"
        a += f"The answer is {self.gold_str(ex)}."
        return QA_TEMPLATE.format(q=q, a=a)

    def build_fewshot(self, n_shots: int) -> str:
        shots = self.fewshot_examples[:n_shots]
        if not shots:
            return ""
        return "\n\n".join(self.format_example(s, True) for s in shots) + "\n\n"

    # ---- scoring
    def extract(self, gen_text: str) -> str:
        span = last_answer_span(gen_text)
        if self.answer_type == "mc":
            return normalize_choice(span)
        if self.answer_type == "number":
            return normalize_number(span)
        return span.strip()

    def is_correct(self, pred: str, ex: dict) -> bool:
        if self.answer_type == "mc":
            return pred == ex["answer"]
        if self.answer_type == "number":
            return numbers_equal(pred, normalize_number(str(ex["answer"])))
        golds = ex.get("golds") or [ex["answer"]]
        if any(normalize_text(pred) == normalize_text(g) for g in golds):
            return True
        return any(numbers_equal(normalize_number(pred), normalize_number(g))
                   for g in golds if re.search(r"\d", g))

    # ---- run
    def evaluate(self, generate_fn, logger, out_dir, *, limit, n_shots,
                 max_new_tokens, temperature, top_p, batch_size):
        instr = self.instruction()
        prefix = self.build_fewshot(n_shots)
        ex_list = self.eval_examples[:limit] if limit else self.eval_examples
        os.makedirs(out_dir, exist_ok=True)
        rec_path = os.path.join(out_dir, f"{self.name}_records.jsonl")
        correct, f1_sum, seen = 0, 0.0, 0

        import json
        with open(rec_path, "w", encoding="utf-8") as rf:
            for batch in chunks(ex_list, batch_size):
                prompts = []
                for ex in batch:
                    full = f"{instr}\n\n{prefix}{self.format_example(ex, False)}"
                    prompts.append(self.codec.encode_text(full)[0] if self.encoded else full)
                gens = generate_fn(prompts, max_new_tokens, temperature, top_p)
                for ex, gen in zip(batch, gens):
                    g = self.codec.decode_text(gen) if self.encoded else gen
                    pred = self.extract(g)
                    ok = self.is_correct(pred, ex)
                    correct += int(ok)
                    seen += 1
                    if self.answer_type == "text":
                        golds = ex.get("golds") or [ex["answer"]]
                        f1_sum += max(token_f1(pred, gg) for gg in golds)
                    rf.write(json.dumps({
                        "question": ex["question"][:600], "gold": self.gold_str(ex),
                        "pred": pred, "correct": bool(ok), "raw": gen[:800],
                    }, ensure_ascii=False) + "\n")
                logger.info(f"[{self.name}] {seen}/{len(ex_list)} "
                            f"acc={correct / max(1, seen):.4f}")

        summary = {"benchmark": self.name, "n": seen, "n_shots": n_shots,
                   "accuracy": correct / max(1, seen)}
        if self.answer_type == "text":
            summary["f1"] = f1_sum / max(1, seen)
        save_json(summary, os.path.join(out_dir, f"{self.name}_summary.json"))
        return summary
