"""TAT-QA (table + text financial QA).

TAT-QA is not published under a single canonical HF id, and rows group several
questions under one table. Set the dataset id in the config, e.g.:

    eval:
      dataset_overrides: {tatqa: "your-org/tat-qa"}

This loader is defensive: it accepts either one-question-per-row schemas or the
original "questions: [...]" grouped schema, and flattens the table + paragraphs
into the context.
"""

from .base import Benchmark


def _flatten_table(table):
    if not table:
        return ""
    if isinstance(table, dict):
        table = table.get("table") or table.get("cells") or []
    if isinstance(table, list):
        return "\n".join(" | ".join(str(c) for c in row) for row in table)
    return str(table)


def _paragraphs(row):
    paras = row.get("paragraphs") or row.get("text") or []
    if isinstance(paras, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paras)
    return str(paras)


def _answer_str(a):
    if isinstance(a, list):
        return ", ".join(str(x) for x in a)
    return str(a)


class TATQA(Benchmark):
    name = "tatqa"
    answer_type = "text"            # answers may be numeric, spans, or lists
    DEFAULT_DATASET = ""            # must be provided via dataset_overrides
    EVAL_SPLIT = "test"
    FEWSHOT_SPLIT = "train"

    def _expand(self, row):
        """Yield (question, answer, context) for every question in a raw row."""
        context = "\n".join(filter(None, [_paragraphs(row),
                                          _flatten_table(row.get("table"))]))
        questions = row.get("questions")
        items = questions if isinstance(questions, list) else [row]
        for qd in items:
            q = qd.get("question") or ""
            ans = qd.get("answer")
            if q and ans is not None:
                yield q.strip(), _answer_str(ans).strip(), context

    def _flatten(self, raw):
        out = []
        for row in raw:
            for q, ans, context in self._expand(row):
                question = f"{context}\n\nQuestion: {q}" if context else q
                out.append({"question": question, "answer": ans, "golds": [ans]})
        return out

    def normalize(self, raw):        # unused (we override load), kept for the ABC
        return None

    def load(self):
        if not self.dataset_id:
            raise ValueError(
                "No dataset id for TAT-QA. Set eval.dataset_overrides.tatqa "
                "to a HF dataset id in your config.")
        self.eval_examples = self._flatten(self._load(self.EVAL_SPLIT))
        try:
            self.fewshot_examples = self._flatten(self._load(self.FEWSHOT_SPLIT))
        except Exception:
            self.fewshot_examples = self.eval_examples[:self.FEWSHOT_POOL]
            self.eval_examples = self.eval_examples[self.FEWSHOT_POOL:]
        return self
