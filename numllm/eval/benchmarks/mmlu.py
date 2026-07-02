from .base import Benchmark


class MMLU(Benchmark):
    name = "mmlu"
    answer_type = "mc"
    DEFAULT_DATASET = "cais/mmlu"
    DATASET_CONFIG = "all"
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"           # 5 curated exemplars per subject

    def normalize(self, e):
        choices = e.get("choices")
        ans = e.get("answer")
        if not choices or ans is None:
            return None
        return {"question": e["question"].strip(),
                "choices": list(choices),
                "answer": chr(65 + int(ans))}
