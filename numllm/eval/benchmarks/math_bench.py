import re

from .base import Benchmark


class MATH(Benchmark):
    name = "math"
    # answers are often expressions (\frac, \sqrt, ...) so we score with
    # normalized-text EM plus a numeric fallback rather than pure numeric.
    answer_type = "text"
    DEFAULT_DATASET = "HuggingFaceH4/MATH-500"     # ungated 500-problem test set
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = None                            # split off the eval set

    def normalize(self, e):
        q = e.get("problem") or e.get("question")
        ans = e.get("answer")
        sol = e.get("solution", "")
        if ans is None:
            m = re.search(r"\\boxed\{(.+?)\}", sol)
            ans = m.group(1) if m else None
        if q is None or ans is None:
            return None
        return {"question": q.strip(), "answer": str(ans).strip(),
                "golds": [str(ans).strip()], "cot": sol or None}
