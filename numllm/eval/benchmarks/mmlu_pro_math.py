import re

from .base import Benchmark


class MMLUProMath(Benchmark):
    name = "mmlu_pro_math"
    answer_type = "mc"
    DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"

    def normalize(self, e):
        if (e.get("category") or "").lower() != "math":
            return None
        opts = e.get("options") or e.get("choices")
        ans = e.get("answer")
        if not opts or not ans:
            return None
        cot = e.get("cot_content") or None
        if cot:
            cot = re.sub(r"[Tt]he answer is.*", "", cot, flags=re.DOTALL).strip()
            cot = re.sub(r"^A:\s*", "", cot).strip() or None
        return {"question": e["question"].strip(), "choices": list(opts),
                "answer": str(ans).strip().upper(), "cot": cot}
