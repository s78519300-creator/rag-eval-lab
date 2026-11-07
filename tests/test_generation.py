from __future__ import annotations

import json

import pytest

from rag_eval_lab.generation import (
    NO_ANSWER,
    ExtractiveGenerator,
    OpenAICompatibleGenerator,
    build_generator,
)


def test_extractive_answer_has_citations(pipeline):
    answer = pipeline.ask("How long do I have to request a refund on a monthly plan?")
    assert "14 days" in answer.text
    assert answer.citations
    assert all(c.startswith("refund-policy.md#chunk") for c in answer.citations)
    for citation in answer.citations:
        assert f"[{citation}]" in answer.text


def test_extractive_declines_when_nothing_matches(pipeline):
    answer = pipeline.ask("What is the capital of France?")
    assert answer.text == NO_ANSWER
    assert answer.citations == [] and answer.is_empty


def test_extractive_respects_max_sentences(pipeline):
    generator = ExtractiveGenerator(max_sentences=1)
    results = pipeline.retrieve("refund monthly annual plan")
    answer = generator.generate("refund monthly annual plan", results)
    assert answer.text.count("#chunk") == 1


def test_extractive_rejects_bad_config():
    with pytest.raises(ValueError):
        ExtractiveGenerator(max_sentences=0)


def test_answer_to_dict_is_json_serialisable(pipeline):
    data = pipeline.ask("When do API tokens expire?").to_dict()
    assert set(data) == {"question", "answer", "citations", "retrieved"}
    json.dumps(data)


def test_build_generator():
    assert isinstance(build_generator("extractive"), ExtractiveGenerator)
    with pytest.raises(ValueError):
        build_generator("magic")


def test_openai_generator_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_LAB_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("RAG_LAB_MODEL", "llama3.1")
    monkeypatch.setenv("RAG_LAB_API_KEY", "test-key")
    gen = OpenAICompatibleGenerator()
    assert gen.base_url == "http://localhost:11434/v1"
    assert gen.model == "llama3.1"
    assert gen.api_key == "test-key"


def test_openai_generator_request_and_citation_parsing(pipeline):
    captured = {}

    def fake_transport(url, headers, body, timeout):
        captured.update(url=url, headers=headers, payload=json.loads(body))
        return {
            "choices": [{"message": {"content": (
                "Data is encrypted with AES-256. [security.md#chunk0] "
                "Also see [made-up.md#chunk9]."
            )}}]
        }

    gen = OpenAICompatibleGenerator(
        base_url="http://llm.local/v1", model="m", api_key="k", transport=fake_transport
    )
    results = pipeline.retrieve("How is customer data encrypted at rest?")
    answer = gen.generate("How is customer data encrypted at rest?", results)

    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["payload"]["model"] == "m"
    assert f"[{results[0].chunk_id}]" in captured["payload"]["messages"][1]["content"]
    # Hallucinated citations are kept so the evaluator can penalise them.
    assert answer.citations == ["security.md#chunk0", "made-up.md#chunk9"]


def test_openai_generator_bad_response_raises(pipeline):
    gen = OpenAICompatibleGenerator(transport=lambda *a: {"unexpected": True})
    with pytest.raises(RuntimeError):
        gen.generate("encryption", pipeline.retrieve("encryption"))
