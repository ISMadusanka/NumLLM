import re

from .base import Benchmark


class GSM8K(Benchmark):
    name = "gsm8k"
    answer_type = "number"
    DEFAULT_DATASET = "openai/gsm8k"
    DATASET_CONFIG = "main"
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "train"

    def normalize(self, e):
        ans = e["answer"]
        final = ans.split("####")[-1].strip().replace(",", "")
        cot = re.sub(r"####.*", "", ans, flags=re.DOTALL).strip()
        cot = re.sub(r"<<.*?>>", "", cot)          # strip calculator annotations
        return {"question": e["question"].strip(), "answer": final, "cot": cot}
