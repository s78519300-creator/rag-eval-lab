"""Evaluation harness for retrieval and answer quality.

Metric definitions (all in ``[0, 1]``, higher is better):

Retrieval - relevance is judged at the *document* level, because eval sets
are much easier to write and maintain that way than per-chunk labels.

* ``recall@k`` - share of relevant documents that appear in the top-k chunks.
* ``precision@k`` - share of the top-k slots filled by chunks from relevant
  documents (empty slots count as misses).
* ``mrr`` - reciprocal rank of the first chunk from a relevant document.

Answer

* ``keyword_coverage`` - share of expected keywords present in the answer.
* ``citation_correctness`` - share of cited chunks that belong to a relevant
  document. An answer with no citations scores 0.
* ``groundedness`` - share of answer sentences that are supported by a
  retrieved chunk (at least ``support_threshold`` of the sentence's content
  words appear in one chunk).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .generation import Answer
from .ingest import PathLike
from .retrieval import RetrievalResult
from .text import split_sentences, strip_citations, tokenize

METRICS = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "keyword_coverage",
    "citation_correctness",
    "groundedness",
)

_SHORT_NAMES = {
    "recall_at_k": "R@k",
    "precision_at_k": "P@k",
    "mrr": "MRR",
    "keyword_coverage": "Keywords",
    "citation_correctness": "Citations",
    "groundedness": "Grounded",
}


@dataclass(frozen=True)
class EvalExample:
    """One labelled question from the eval set."""

    id: str
    question: str
    expected_keywords: List[str]
    relevant_docs: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_id: str) -> "EvalExample":
        missing = [key for key in ("question", "relevant_docs") if key not in data]
        if missing:
            raise ValueError(f"Eval example {default_id} is missing fields: {', '.join(missing)}")
        return cls(
            id=str(data.get("id", default_id)),
            question=str(data["question"]),
            expected_keywords=[str(k) for k in data.get("expected_keywords", [])],
            relevant_docs=[str(d) for d in data["relevant_docs"]],
        )


def load_evalset(path: PathLike) -> List[EvalExample]:
    """Load a JSONL eval set. Blank lines and ``//`` comment lines are skipped."""
    examples: List[EvalExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            examples.append(EvalExample.from_dict(data, default_id=f"q{len(examples) + 1}"))
    if not examples:
        raise ValueError(f"Eval set {path} contains no examples")
    return examples


# --- retrieval metrics ------------------------------------------------------


def recall_at_k(retrieved_docs: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of relevant documents found among the first ``k`` retrieved items."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(relevant_set & set(retrieved_docs[:k])) / len(relevant_set)


def precision_at_k(retrieved_docs: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the first ``k`` retrieved items that come from relevant documents."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for doc in retrieved_docs[:k] if doc in relevant_set) / k


def reciprocal_rank(retrieved_docs: Sequence[str], relevant: Sequence[str]) -> float:
    """``1 / rank`` of the first relevant item, or 0 if none was retrieved."""
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


# --- answer metrics ---------------------------------------------------------


def keyword_coverage(answer_text: str, keywords: Sequence[str]) -> float:
    """Share of ``keywords`` that appear (case-insensitively) in the answer."""
    if not keywords:
        return 1.0
    text = strip_citations(answer_text).lower()
    return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)


def citation_correctness(citations: Sequence[str], relevant: Sequence[str]) -> float:
    """Share of cited chunk ids whose document is in ``relevant``."""
    if not citations:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for c in citations if c.split("#", 1)[0] in relevant_set) / len(citations)


def groundedness(
    answer_text: str, results: Sequence[RetrievalResult], support_threshold: float = 0.7
) -> float:
    """Share of answer sentences supported by at least one retrieved chunk."""
    sentences = split_sentences(strip_citations(answer_text))
    if not sentences:
        return 0.0
    chunk_terms = [set(tokenize(r.chunk.text)) for r in results]
    supported = 0
    for sentence in sentences:
        terms = set(tokenize(sentence))
        if not terms:
            continue
        best = max((len(terms & ct) / len(terms) for ct in chunk_terms), default=0.0)
        if best >= support_threshold:
            supported += 1
    return supported / len(sentences)


# --- harness ----------------------------------------------------------------


@dataclass
class ExampleResult:
    """Per-question metrics plus the artefacts needed to debug a failure."""

    id: str
    question: str
    answer: str
    citations: List[str]
    retrieved: List[str]
    metrics: Dict[str, float]


@dataclass
class EvalReport:
    """Aggregated results of an evaluation run."""

    k: int
    results: List[ExampleResult]
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Mean of every metric across all examples."""
        n = len(self.results)
        return {
            name: round(sum(r.metrics[name] for r in self.results) / n, 4) if n else 0.0
            for name in METRICS
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "num_examples": len(self.results),
            "config": self.config,
            "summary": self.summary(),
            "examples": [asdict(r) for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def _rows(self) -> List[List[str]]:
        rows = [[r.id] + [f"{r.metrics[m]:.2f}" for m in METRICS] for r in self.results]
        summary = self.summary()
        rows.append(["mean"] + [f"{summary[m]:.2f}" for m in METRICS])
        return rows

    def _headers(self) -> List[str]:
        return ["id"] + [_SHORT_NAMES[m].replace("@k", f"@{self.k}") for m in METRICS]

    def format_table(self) -> str:
        """Fixed-width text table for terminal output."""
        headers = self._headers()
        rows = self._rows()
        widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]

        def fmt(row: List[str]) -> str:
            first = row[0].ljust(widths[0])
            rest = [cell.rjust(w) for cell, w in zip(row[1:], widths[1:])]
            return "  ".join([first] + rest)

        rule = "-" * len(fmt(headers))
        lines = [fmt(headers), rule] + [fmt(r) for r in rows[:-1]] + [rule, fmt(rows[-1])]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Markdown report with a summary, a per-question table and failures."""
        headers = self._headers()
        summary = self.summary()
        lines = [
            "# RAG evaluation report",
            "",
            f"- Examples: **{len(self.results)}**",
            f"- Retrieval depth (k): **{self.k}**",
        ]
        for key, value in self.config.items():
            lines.append(f"- {key.replace('_', ' ').capitalize()}: `{value}`")
        lines += ["", "## Summary", "", "| Metric | Score |", "| --- | ---: |"]
        lines += [f"| {name} | {summary[name]:.3f} |" for name in METRICS]
        lines += [
            "",
            "## Per-question results",
            "",
            "| " + " | ".join(headers) + " |",
            "| --- | " + " | ".join("---:" for _ in headers[1:]) + " |",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in self._rows()[:-1]]

        weak = [r for r in self.results if min(r.metrics.values()) < 0.5]
        lines += ["", "## Questions to review", ""]
        if not weak:
            lines.append("Every question scored at least 0.5 on every metric.")
        for r in weak:
            low = ", ".join(f"{m}={v:.2f}" for m, v in r.metrics.items() if v < 0.5)
            lines += [
                f"### {r.id}: {r.question}",
                "",
                f"- Low metrics: {low}",
                f"- Retrieved: {', '.join(r.retrieved) or '(nothing)'}",
                f"- Answer: {r.answer}",
                "",
            ]
        return "\n".join(lines).rstrip() + "\n"


def evaluate_example(
    example: EvalExample, answer: Answer, k: int, support_threshold: float = 0.7
) -> ExampleResult:
    """Score a single answer against its labelled example."""
    retrieved_docs = [r.doc_id for r in answer.retrieved]
    metrics = {
        "recall_at_k": recall_at_k(retrieved_docs, example.relevant_docs, k),
        "precision_at_k": precision_at_k(retrieved_docs, example.relevant_docs, k),
        "mrr": reciprocal_rank(retrieved_docs[:k], example.relevant_docs),
        "keyword_coverage": keyword_coverage(answer.text, example.expected_keywords),
        "citation_correctness": citation_correctness(answer.citations, example.relevant_docs),
        "groundedness": groundedness(answer.text, answer.retrieved, support_threshold),
    }
    return ExampleResult(
        id=example.id,
        question=example.question,
        answer=answer.text,
        citations=list(answer.citations),
        retrieved=[r.chunk_id for r in answer.retrieved],
        metrics={name: round(value, 4) for name, value in metrics.items()},
    )


def run_evaluation(
    pipeline: Any,
    examples: Sequence[EvalExample],
    k: Optional[int] = None,
    support_threshold: float = 0.7,
) -> EvalReport:
    """Ask every question through ``pipeline`` and score the answers.

    ``pipeline`` is anything with an ``ask(question, k=...)`` method returning
    an :class:`~rag_eval_lab.generation.Answer` and a ``top_k`` attribute,
    which keeps the harness usable with custom pipelines.
    """
    depth = k or pipeline.top_k
    results = [
        evaluate_example(ex, pipeline.ask(ex.question, k=depth), depth, support_threshold)
        for ex in examples
    ]
    config = {"generator": getattr(getattr(pipeline, "generator", None), "name", "custom")}
    return EvalReport(k=depth, results=results, config=config)
