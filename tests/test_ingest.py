from __future__ import annotations

import pytest

from rag_eval_lab.ingest import Document, chunk_document, chunk_documents, load_documents
from rag_eval_lab.text import split_sentences, strip_citations, tokenize


def test_load_documents_reads_md_and_txt_only(tiny_corpus):
    docs = load_documents(tiny_corpus)
    assert [d.doc_id for d in docs] == ["cats.md", "dogs.txt", "rockets.md"]


def test_load_documents_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_load_single_file(tiny_corpus):
    docs = load_documents(tiny_corpus / "dogs.txt")
    assert len(docs) == 1 and docs[0].doc_id == "dogs.txt"


def test_chunking_respects_size_and_overlap():
    words = [f"w{i}" for i in range(25)]
    doc = Document(doc_id="d.md", text=" ".join(words))
    chunks = chunk_document(doc, chunk_size=10, overlap=3)

    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[2].chunk_id == "d.md#chunk2"
    assert all(len(c.text.split()) <= 10 for c in chunks)
    # Last 3 words of one chunk are the first 3 words of the next.
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w24"


def test_chunking_preserves_line_breaks():
    doc = Document(doc_id="d.md", text="# Title\n\nFirst line here.\nSecond line here.")
    (chunk,) = chunk_document(doc, chunk_size=50, overlap=0)
    assert chunk.text == doc.text


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_chunking_rejects_bad_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document("d.md", "a b c"), chunk_size=size, overlap=overlap)


def test_example_corpus_chunks(example_docs):
    chunks = chunk_documents(load_documents(example_docs))
    assert len({c.doc_id for c in chunks}) == 5
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The refunds are processed") == ["refund", "processed"]
    assert tokenize("99.95% uptime") == ["99.95", "uptime"]


def test_split_sentences_drops_headings_and_bullets():
    text = "## Heading\n\n- Bullet point with words.\nOne sentence here. Another one follows!"
    assert split_sentences(text) == [
        "Bullet point with words.",
        "One sentence here.",
        "Another one follows!",
    ]


def test_strip_citations():
    assert strip_citations("Fact one. [a.md#chunk0] Fact two. [b.md#chunk3]") == (
        "Fact one. Fact two."
    )
