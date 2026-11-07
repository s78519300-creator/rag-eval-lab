from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_lab import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DOCS = ROOT / "examples" / "docs"
EXAMPLE_EVALSET = ROOT / "examples" / "evalset.jsonl"


@pytest.fixture
def example_docs() -> Path:
    return EXAMPLE_DOCS


@pytest.fixture
def example_evalset() -> Path:
    return EXAMPLE_EVALSET


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline.from_path(EXAMPLE_DOCS)


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """A three-document corpus small enough to reason about by hand."""
    (tmp_path / "cats.md").write_text(
        "# Cats\n\nCats are small domestic animals. Cats sleep for most of the day.\n",
        encoding="utf-8",
    )
    (tmp_path / "dogs.txt").write_text(
        "Dogs are loyal companions. Dogs need a daily walk outside.\n", encoding="utf-8"
    )
    (tmp_path / "rockets.md").write_text(
        "Rockets burn liquid fuel to reach orbit. Orbit requires high velocity.\n",
        encoding="utf-8",
    )
    (tmp_path / "ignored.csv").write_text("not,a,document\n", encoding="utf-8")
    return tmp_path
