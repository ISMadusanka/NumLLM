from .base import Benchmark


class SVAMP(Benchmark):
    name = "svamp"
    answer_type = "number"
    DEFAULT_DATASET = "ChilleD/SVAMP"
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "train"

    def normalize(self, e):
        body = e.get("Body") or e.get("body") or ""
        q = e.get("Question") or e.get("question") or ""
        ans = e.get("Answer", e.get("answer"))
        if ans is None:
            return None
        if isinstance(ans, float) and ans.is_integer():
            ans_str = str(int(ans))
        else:
            ans_str = str(ans)
        eq = e.get("Equation") or e.get("equation")
        cot = f"Compute {eq}." if eq else None
        return {"question": f"{body} {q}".strip(), "answer": ans_str, "cot": cot}
