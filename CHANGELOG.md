# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-08

### Added
- Document loading for `.md` and `.txt` files with word-window chunking
  (configurable size and overlap) and stable `doc#chunkN` ids.
- Pure-Python BM25 retriever with deterministic top-k ranking.
- `ExtractiveGenerator`: offline, deterministic answers with inline citations.
- `OpenAICompatibleGenerator`: optional generator for any OpenAI-compatible
  chat completions endpoint, configured through `RAG_LAB_*` environment variables.
- Evaluation harness: recall@k, precision@k, MRR, keyword coverage, citation
  correctness and groundedness, with terminal, JSON and Markdown reports.
- `--fail-under` quality gates for CI.
- Minimal REST API (`POST /ask`, `GET /health`) on the standard library.
- `rag-lab` CLI with `ingest`, `ask`, `eval` and `serve` commands.
- Northwind Cloud example corpus and an 8-question eval set.
