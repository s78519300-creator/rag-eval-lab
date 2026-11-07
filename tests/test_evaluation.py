from __future__ import annotations

import json

import pytest

from rag_eval_lab.evaluation import (
    METRICS,
    citation_correctness,
    groundedness,
    keyword_coverage,
    load_evalset,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
)
from rag_eval_lab.ingest import Chunk
from rag_eval_lab.retrieval import RetrievalResult


def test_retrieval_metrics():
    retrieved = ["b.md", "a.md", "a.md"]
    assert recall_at_k(retrieved, ["a.md", "c.md"], k=3) == 0.5
    assert recall_at_k(retrieved, ["a.md"], k=1) == 0.0
    assert precision_at_k(retrieved, ["a.md"], k=3) == pytest.approx(2 / 3)
    assert precision_at_k(["a.md"], ["a.md"], k=3) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["a.md"]) == 0.5
    assert reciprocal_rank(retrieved, ["z.md"]) == 0.0


def test_keyword_coverage_is_case_insensitive():
    assert keyword_coverage("Refunds within 14 Days. [x.md#chunk0]", ["14 days", "AES"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0


def test_citation_correctness():
    assert citation_correctness(["a.md#chunk0", "b.md#chunk1"], ["a.md"]) == 0.5
    assert citation_correctness([], ["a.md"]) == 0.0


def test_groundedness_detects_unsupported_sentences():
    chunk = Chunk("a.md", 0, "Data is encrypted at rest with AES-256 keys.")
    results = [RetrievalResult(chunk=chunk, score=1.0, rank=1)]
    grounded = "Data is encrypted at rest with AES-256. [a.md#chunk0]"
    mixed = grounded + " The moon is made of green cheese entirely."
    assert groundedness(grounded, results) == 1.0
    assert groundedness(mixed, results) == 0.5
    assert groundedness("", results) == 0.0


def test_load_evalset(example_evalset):
    examples = load_evalset(example_evalset)
    assert len(examples) == 8
    assert examples[0].relevant_docs == ["refund-policy.md"]


def test_load_evalset_reports_bad_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "q", "relevant_docs": []}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_evalset(path)
    path.write_text('{"relevant_docs": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields: question"):
        load_evalset(path)


def test_full_evaluation_on_examples(pipeline, example_evalset):
    report = run_evaluation(pipeline, load_evalset(example_evalset))
    summary = report.summary()
    assert set(summary) == set(METRICS)
    assert all(0.0 <= v <= 1.0 for v in summary.values())
    # The example corpus is small and clean, so retrieval should be near perfect.
    assert summary["recall_at_k"] >= 0.9
    assert summary["mrr"] >= 0.9
    assert summary["groundedness"] >= 0.9


def test_report_formats(pipeline, example_evalset):
    report = run_evaluation(pipeline, load_evalset(example_evalset), k=2)
    data = json.loads(report.to_json())
    assert data["k"] == 2 and data["num_examples"] == 8
    assert "R@2" in report.format_table()
    markdown = report.to_markdown()
    assert markdown.startswith("# RAG evaluation report")
    assert "| recall_at_k |" in markdown
