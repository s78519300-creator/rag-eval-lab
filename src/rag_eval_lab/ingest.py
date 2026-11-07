"""Document loading and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

SUPPORTED_SUFFIXES = (".md", ".txt")

_WORD_SPAN_RE = re.compile(r"\S+")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Document:
    """A source document loaded from disk.

    ``doc_id`` is the path relative to the corpus root, using forward slashes,
    so that ids are stable across operating systems.
    """

    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document that the retriever indexes."""

    doc_id: str
    index: int
    text: str

    @property
    def chunk_id(self) -> str:
        """Citation-friendly identifier, e.g. ``refund-policy.md#chunk2``."""
        return f"{self.doc_id}#chunk{self.index}"


def load_documents(root: PathLike) -> List[Document]:
    """Load every ``.md`` / ``.txt`` file under ``root`` (recursively).

    ``root`` may also point at a single file. Documents are returned sorted by
    id so that chunk numbering and retrieval ties are deterministic.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {root_path}")

    if root_path.is_file():
        files = [root_path]
        base = root_path.parent
    else:
        files = [
            p for p in root_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        base = root_path

    documents = []
    for path in sorted(files):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(doc_id=path.relative_to(base).as_posix(), text=text))
    return documents


def chunk_document(document: Document, chunk_size: int = 120, overlap: int = 20) -> List[Chunk]:
    """Split a document into overlapping word-window chunks.

    Windows are measured in whitespace-separated words, but each chunk keeps
    the *original* text between its first and last word, so line breaks and
    markdown structure survive. That matters later when the generator splits
    chunks into sentences.

    Args:
        document: The document to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared by consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    spans = [m.span() for m in _WORD_SPAN_RE.finditer(document.text)]
    if not spans:
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    start = 0
    while True:
        end = min(start + chunk_size, len(spans))
        text = document.text[spans[start][0] : spans[end - 1][1]]
        chunks.append(Chunk(doc_id=document.doc_id, index=len(chunks), text=text))
        if end == len(spans):
            break
        start += step
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 120, overlap: int = 20
) -> List[Chunk]:
    """Chunk several documents, preserving document order."""
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
