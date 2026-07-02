from .base import Benchmark


class DROP(Benchmark):
    name = "drop"
    answer_type = "text"            # scored with EM + F1 (+ numeric fallback)
    DEFAULT_DATASET = "ucinlp/drop"
    EVAL_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"

    def normalize(self, e):
        spans = (e.get("answers_spans") or {}).get("spans") or []
        spans = [s for s in spans if str(s).strip()]
        if not spans:
            return None
        q = (f"Passage:\n{e['passage'].strip()}\n\n"
             f"Question: {e['question'].strip()}")
        return {"question": q, "answer": spans[0], "golds": [str(s) for s in spans]}
