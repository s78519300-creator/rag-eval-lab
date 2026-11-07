"""A small, readable Okapi BM25 implementation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .ingest import Chunk
from .text import tokenize


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk together with its BM25 score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class BM25Index:
    """In-memory BM25 index over a list of chunks.

    Uses the common "BM25+"-style non-negative IDF,
    ``log(1 + (N - df + 0.5) / (df + 0.5))``, so that terms appearing in most
    chunks still contribute a small positive weight instead of a negative one.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.chunks: List[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b

        self._term_freqs: List[Counter] = [Counter(tokenize(c.text)) for c in self.chunks]
        self._lengths: List[int] = [sum(tf.values()) for tf in self._term_freqs]
        n = len(self.chunks)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0

        doc_freq: Counter = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        self._idf: Dict[str, float] = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term`` (0.0 if unseen)."""
        return self._idf.get(term, 0.0)

    def score(self, query: str, position: int) -> float:
        """BM25 score of the chunk at ``position`` for ``query``."""
        tf = self._term_freqs[position]
        length = self._lengths[position]
        norm = self.k1 * (1 - self.b + self.b * length / self._avg_len) if self._avg_len else 0.0
        total = 0.0
        for term in tokenize(query):
            freq = tf.get(term, 0)
            if freq:
                total += self.idf(term) * freq * (self.k1 + 1) / (freq + norm)
        return total

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """Return the ``k`` best-scoring chunks for ``query``.

        Chunks with a score of zero (no query term overlap) are never returned.
        Ties are broken by corpus order, which keeps results deterministic.
        """
        if k <= 0:
            return []
        scored = [(self.score(query, i), i) for i in range(len(self.chunks))]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalResult(chunk=self.chunks[i], score=score, rank=rank)
            for rank, (score, i) in enumerate(scored[:k], start=1)
        ]
