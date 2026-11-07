"""Command-line interface: ``rag-lab ingest | ask | eval | serve``."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from . import __version__
from .evaluation import METRICS, load_evalset, run_evaluation
from .generation import build_generator
from .pipeline import RAGPipeline


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs", required=True, help="Directory (or file) of .md/.txt documents")
    parser.add_argument("--chunk-size", type=int, default=120, help="Words per chunk (default 120)")
    parser.add_argument("--overlap", type=int, default=20, help="Words of overlap (default 20)")


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve (default 3)")
    parser.add_argument(
        "--generator",
        choices=("extractive", "openai"),
        default="extractive",
        help="Answer generator (default: offline extractive)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-lab",
        description="Document Q&A with cited answers, plus an evaluation harness.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load and chunk documents, then print corpus stats")
    _add_corpus_args(ingest)

    ask = sub.add_parser("ask", help="Answer a question with citations")
    ask.add_argument("question", help="The question to ask")
    _add_corpus_args(ask)
    _add_generation_args(ask)
    ask.add_argument("--json", action="store_true", help="Print the full answer as JSON")
    ask.add_argument("--show-context", action="store_true", help="Also print retrieved chunks")

    ev = sub.add_parser("eval", help="Run the evaluation harness over a JSONL eval set")
    _add_corpus_args(ev)
    _add_generation_args(ev)
    ev.add_argument("--evalset", required=True, help="Path to the JSONL eval set")
    ev.add_argument("--json-out", help="Write the full report as JSON to this path")
    ev.add_argument("--md-out", help="Write a Markdown report to this path")
    ev.add_argument(
        "--fail-under",
        metavar="METRIC=VALUE",
        action="append",
        default=[],
        help="Exit non-zero if a mean metric is below VALUE, e.g. recall_at_k=0.8 (repeatable)",
    )

    srv = sub.add_parser("serve", help="Start the REST API (POST /ask)")
    _add_corpus_args(srv)
    _add_generation_args(srv)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)

    return parser


def _pipeline_from_args(args: argparse.Namespace) -> RAGPipeline:
    generator = build_generator(getattr(args, "generator", "extractive"))
    return RAGPipeline.from_path(
        args.docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        generator=generator,
        top_k=getattr(args, "top_k", 3),
    )


def _parse_thresholds(specs: List[str]) -> dict:
    thresholds = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        if not sep or name not in METRICS:
            raise ValueError(f"Invalid --fail-under {spec!r}; metrics: {', '.join(METRICS)}")
        thresholds[name] = float(value)
    return thresholds


def cmd_ingest(args: argparse.Namespace) -> int:
    stats = _pipeline_from_args(args).stats()
    print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks "
          f"(avg {stats['avg_words_per_chunk']} words/chunk)")
    for doc_id, count in stats["chunks_per_document"].items():
        print(f"  {doc_id:<28} {count} chunk{'s' if count != 1 else ''}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    answer = _pipeline_from_args(args).ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    print(f"Q: {answer.question}\n")
    print(textwrap.fill(answer.text, width=88, initial_indent="A: ", subsequent_indent="   "))
    print("\nSources:")
    for result in answer.retrieved:
        marker = "*" if result.chunk_id in answer.citations else " "
        print(f" {marker} {result.rank}. {result.chunk_id:<28} score={result.score:.3f}")
        if args.show_context:
            print(textwrap.indent(textwrap.fill(result.chunk.text, width=80), "      "))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    thresholds = _parse_thresholds(args.fail_under)
    pipeline = _pipeline_from_args(args)
    report = run_evaluation(pipeline, load_evalset(args.evalset))
    print(f"Evaluated {len(report.results)} questions (k={report.k}, "
          f"generator={report.config['generator']})\n")
    print(report.format_table())

    outputs = [(args.json_out, report.to_json() + "\n"), (args.md_out, report.to_markdown())]
    written = []
    for path, content in outputs:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            written.append(path)
    if written:
        print("\nWrote " + ", ".join(written))

    summary = report.summary()
    failures = [f"{m}={summary[m]:.3f} < {v}" for m, v in thresholds.items() if summary[m] < v]
    if failures:
        print("\nQuality gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    serve(_pipeline_from_args(args), host=args.host, port=args.port)
    return 0


COMMANDS = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "serve": cmd_serve}


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``rag-lab`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"rag-lab: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
