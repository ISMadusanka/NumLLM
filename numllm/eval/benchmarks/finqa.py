from .base import Benchmark


def _join(x):
    if isinstance(x, list):
        return " ".join(str(t) for t in x)
    return str(x or "")


def _flatten_table(table):
    if not table:
        return ""
    if isinstance(table, list):
        return "\n".join(" | ".join(str(c) for c in row) for row in table)
    return str(table)


class FinQA(Benchmark):
    name = "finqa"
    answer_type = "number"          # financial answers are mostly numeric
    DEFAULT_DATASET = "dreamerdeo/finqa"
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "train"

    def normalize(self, e):
        q = e.get("question") or ""
        ans = e.get("answer")
        if ans is None:
            ans = e.get("exe_ans")
        if not q or ans is None:
            return None
        context = "\n".join(filter(None, [
            _join(e.get("pre_text")),
            _flatten_table(e.get("table")),
            _join(e.get("post_text")),
            f"Question: {q.strip()}",
        ]))
        ans = str(ans).strip()
        return {"question": context.strip(), "answer": ans, "golds": [ans]}
