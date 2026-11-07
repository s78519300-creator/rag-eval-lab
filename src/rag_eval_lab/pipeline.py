"""The retrieve-then-generate pipeline that ties the pieces together."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from .generation import Answer, ExtractiveGenerator, Generator
from .ingest import Chunk, PathLike, chunk_documents, load_documents
from .retrieval import BM25Index, RetrievalResult


class RAGPipeline:
    """Retrieval-augmented question answering over an in-memory corpus.

    Args:
        chunks: Pre-chunked corpus.
        generator: Answer generator; defaults to :class:`ExtractiveGenerator`.
        top_k: Number of chunks to retrieve per question.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        generator: Optional[Generator] = None,
        top_k: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.chunks: List[Chunk] = list(chunks)
        self.index = BM25Index(self.chunks)
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k

    @classmethod
    def from_path(
        cls,
        docs: PathLike,
        chunk_size: int = 120,
        overlap: int = 20,
        generator: Optional[Generator] = None,
        top_k: int = 3,
    ) -> "RAGPipeline":
        """Load, chunk and index every document under ``docs``."""
        documents = load_documents(docs)
        if not documents:
            raise ValueError(f"No .md or .txt documents found under {docs}")
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        return cls(chunks, generator=generator, top_k=top_k)

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """Return the top ``k`` chunks for ``question``."""
        return self.index.search(question, k=k or self.top_k)

    def ask(self, question: str, k: Optional[int] = None) -> Answer:
        """Retrieve context for ``question`` and generate a cited answer."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        return self.generator.generate(question, self.retrieve(question, k=k))

    def stats(self) -> Dict[str, object]:
        """Summary of the indexed corpus, used by ``rag-lab ingest``."""
        per_doc = Counter(c.doc_id for c in self.chunks)
        words = [len(c.text.split()) for c in self.chunks]
        return {
            "documents": len(per_doc),
            "chunks": len(self.chunks),
            "avg_words_per_chunk": round(sum(words) / len(words), 1) if words else 0.0,
            "chunks_per_document": dict(sorted(per_doc.items())),
        }
