"""Answer generators.

Two generators ship with the project:

* :class:`ExtractiveGenerator` - the default. Fully offline and deterministic.
  It picks the retrieved sentences that best overlap the question and cites
  the chunk each sentence came from.
* :class:`OpenAICompatibleGenerator` - calls any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, Azure-style gateways, vLLM, Ollama,
  LM Studio, ...) using only ``urllib``. Configured through environment
  variables so no secrets end up in code or config files.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .retrieval import RetrievalResult
from .text import split_sentences, tokenize

NO_ANSWER = "I could not find an answer to that in the provided documents."

_CITATION_ID_RE = re.compile(r"\[([^\[\]\s]+#chunk\d+)\]")


@dataclass
class Answer:
    """A generated answer plus everything needed to evaluate it."""

    question: str
    text: str
    citations: List[str] = field(default_factory=list)
    retrieved: List[RetrievalResult] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the generator declined to answer."""
        return not self.citations

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation used by the CLI and REST API."""
        return {
            "question": self.question,
            "answer": self.text,
            "citations": list(self.citations),
            "retrieved": [
                {"chunk_id": r.chunk_id, "doc_id": r.doc_id, "rank": r.rank,
                 "score": round(r.score, 4)}
                for r in self.retrieved
            ],
        }


class Generator:
    """Interface for answer generators."""

    name = "base"

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> Answer:
        raise NotImplementedError


def _unique(items: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


class ExtractiveGenerator(Generator):
    """Build an answer from the retrieved sentences that best match the question.

    Each candidate sentence is scored by the share of (stemmed, non-stopword)
    question terms it contains, with a small bonus for coming from a
    higher-ranked chunk. The top ``max_sentences`` sentences are returned,
    each followed by a citation such as ``[sla.md#chunk1]``.

    Args:
        max_sentences: Upper bound on sentences in the answer.
        min_overlap: Minimum share of question terms a sentence must contain
            to be considered at all. Below this the generator says it does
            not know instead of guessing.
        relative_cutoff: Supporting sentences must score at least this
            fraction of the best sentence, which stops weakly related
            sentences from padding the answer.
    """

    name = "extractive"

    def __init__(
        self, max_sentences: int = 2, min_overlap: float = 0.25, relative_cutoff: float = 0.75
    ) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self.relative_cutoff = relative_cutoff

    def score_sentence(self, question_terms: set, sentence: str, rank: int) -> float:
        """Score one sentence against the question terms."""
        if not question_terms:
            return 0.0
        overlap = len(question_terms & set(tokenize(sentence))) / len(question_terms)
        if overlap < self.min_overlap:
            return 0.0
        return overlap + 0.05 / rank

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> Answer:
        question_terms = set(tokenize(question))
        candidates: List[Tuple[float, int, str, str]] = []
        seen_sentences = set()
        order = 0
        for result in results:
            for sentence in split_sentences(result.chunk.text):
                key = sentence.lower()
                if key in seen_sentences:  # overlapping chunks repeat sentences
                    continue
                seen_sentences.add(key)
                score = self.score_sentence(question_terms, sentence, result.rank)
                if score > 0:
                    candidates.append((score, order, sentence, result.chunk_id))
                order += 1

        if not candidates:
            return Answer(question=question, text=NO_ANSWER, retrieved=list(results))

        candidates.sort(key=lambda c: (-c[0], c[1]))
        best = candidates[0][0]
        chosen = [c for c in candidates if c[0] >= best * self.relative_cutoff]
        chosen = chosen[: self.max_sentences]
        parts = [f"{sentence} [{chunk_id}]" for _, _, sentence, chunk_id in chosen]
        citations = _unique([chunk_id for _, _, _, chunk_id in chosen])
        return Answer(
            question=question, text=" ".join(parts), citations=citations, retrieved=list(results)
        )


SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "After every sentence, cite the supporting context block by its id in square "
    "brackets, for example [handbook.md#chunk0]. "
    "If the context does not contain the answer, say you could not find it."
)

Transport = Callable[[str, Dict[str, str], bytes, float], Dict[str, Any]]


def _urllib_transport(
    url: str, headers: Dict[str, str], body: bytes, timeout: float
) -> Dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc


class OpenAICompatibleGenerator(Generator):
    """Generate answers with an OpenAI-compatible chat completions API.

    Configuration (constructor arguments win over environment variables):

    * ``RAG_LAB_BASE_URL`` - API base, default ``https://api.openai.com/v1``
    * ``RAG_LAB_MODEL`` - model name, default ``gpt-4o-mini``
    * ``RAG_LAB_API_KEY`` - bearer token (falls back to ``OPENAI_API_KEY``);
      optional for local servers that do not check it.

    ``transport`` exists so the request/response handling can be unit tested
    without network access.
    """

    name = "openai"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        transport: Optional[Transport] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("RAG_LAB_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("RAG_LAB_MODEL") or "gpt-4o-mini"
        self.api_key = api_key or os.environ.get("RAG_LAB_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self.temperature = temperature
        self.timeout = timeout
        self._transport = transport or _urllib_transport

    @staticmethod
    def build_context(results: Sequence[RetrievalResult]) -> str:
        """Format retrieved chunks as labelled context blocks."""
        return "\n\n".join(f"[{r.chunk_id}]\n{r.chunk.text}" for r in results)

    def build_messages(
        self, question: str, results: Sequence[RetrievalResult]
    ) -> List[Dict[str, str]]:
        """Build the chat messages sent to the endpoint."""
        user = f"Context:\n{self.build_context(results)}\n\nQuestion: {question}"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> Answer:
        if not results:
            return Answer(question=question, text=NO_ANSWER, retrieved=[])

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.build_messages(question, results),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._transport(
            f"{self.base_url}/chat/completions",
            headers,
            json.dumps(payload).encode("utf-8"),
            self.timeout,
        )
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"Unexpected response from LLM endpoint: {data!r:.300}") from exc

        # Keep every citation the model produced, even ones pointing at chunks we
        # never supplied: the evaluator should see (and penalise) those.
        citations = _unique(_CITATION_ID_RE.findall(text))
        return Answer(question=question, text=text, citations=citations, retrieved=list(results))


def build_generator(name: str) -> Generator:
    """Create a generator by name (``extractive`` or ``openai``)."""
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "openai":
        return OpenAICompatibleGenerator()
    raise ValueError(f"Unknown generator: {name!r} (expected 'extractive' or 'openai')")
