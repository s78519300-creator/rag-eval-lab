from __future__ import annotations

import pytest

from rag_eval_lab.ingest import chunk_documents, load_documents
from rag_eval_lab.retrieval import BM25Index


@pytest.fixture
def index(tiny_corpus):
    return BM25Index(chunk_documents(load_documents(tiny_corpus)))


def test_search_ranks_most_relevant_first(index):
    results = index.search("how much do cats sleep", k=3)
    assert results[0].doc_id == "cats.md"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_search_excludes_zero_score_chunks(index):
    assert index.search("quantum chromodynamics") == []
    assert index.search("orbit", k=10)[0].doc_id == "rockets.md"
    assert len(index.search("orbit", k=10)) == 1


def test_search_respects_k(index):
    assert len(index.search("cats dogs rockets", k=2)) == 2
    assert index.search("cats", k=0) == []


def test_idf_rarer_terms_weigh_more(index):
    # "cat" appears in one chunk, unknown terms get zero weight.
    assert index.idf("cat") > 0
    assert index.idf("zebra") == 0.0


def test_scores_are_deterministic(index):
    first = [(r.chunk_id, r.score) for r in index.search("dogs walk", k=3)]
    second = [(r.chunk_id, r.score) for r in index.search("dogs walk", k=3)]
    assert first == second


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Index([], b=1.5)
    with pytest.raises(ValueError):
        BM25Index([], k1=-1)


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything") == []


def test_example_corpus_retrieves_sla_for_uptime(pipeline):
    results = pipeline.retrieve("guaranteed monthly uptime for managed databases")
    assert results[0].doc_id == "sla.md"
