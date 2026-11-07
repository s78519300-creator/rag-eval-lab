"""Small text utilities shared by the retriever, generator and evaluator.

Everything here is deliberately simple and dependency-free. The goal is
predictable behaviour that is easy to reason about in tests, not linguistic
perfection.
"""

from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+|\.[0-9]+)*")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION_RE = re.compile(r"\s*\[[^\[\]]+#chunk\d+\]")
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)")

# A compact English stopword list. Large enough to stop BM25 and sentence
# scoring from being dominated by filler words, small enough to read.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its may me my of on or our should so than that
    the their them then there these they this to was we were what when where
    which who why will with would you your yours about after before any all
    also just more most must not no only other over such up out per via
    """.split()
)


def stem(token: str) -> str:
    """A deliberately tiny plural stripper ("refunds" -> "refund").

    It is not a real stemmer; it only folds the most common English plural
    forms so that queries and documents agree on simple variations.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, stemming: bool = True) -> List[str]:
    """Lowercase ``text`` and split it into word tokens.

    Hyphenated words and contractions are kept together ("multi-factor",
    "customer's"). Stopwords are dropped unless ``remove_stopwords`` is False,
    and simple plurals are folded unless ``stemming`` is False.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def clean_markdown_line(line: str) -> str:
    """Strip leading markdown markers (headings, bullets, quotes) from a line."""
    return _MARKDOWN_PREFIX_RE.sub("", line).replace("**", "").replace("`", "").strip()


def split_sentences(text: str, *, min_tokens: int = 3) -> List[str]:
    """Split ``text`` into sentences.

    Line breaks are treated as hard boundaries so that markdown headings and
    bullet points do not get glued to the following paragraph. Fragments with
    fewer than ``min_tokens`` words (typically headings) are discarded.
    """
    sentences: List[str] = []
    for piece in _SENTENCE_BOUNDARY_RE.split(text):
        cleaned = clean_markdown_line(piece)
        if len(tokenize(cleaned, remove_stopwords=False)) >= min_tokens:
            sentences.append(cleaned)
    return sentences


def strip_citations(text: str) -> str:
    """Remove ``[doc.md#chunkN]`` style citation markers from ``text``."""
    return _CITATION_RE.sub("", text).strip()
