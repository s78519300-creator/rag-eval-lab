from __future__ import annotations

import json
import subprocess
import sys

from rag_eval_lab.cli import main


def test_ingest_command(example_docs, capsys):
    assert main(["ingest", "--docs", str(example_docs)]) == 0
    out = capsys.readouterr().out
    assert "Indexed 5 documents" in out
    assert "refund-policy.md" in out


def test_ask_command_text_and_json(example_docs, capsys):
    assert main(["ask", "When do API tokens expire?", "--docs", str(example_docs)]) == 0
    out = capsys.readouterr().out
    assert "90 days" in out and "Sources:" in out

    assert main(["ask", "When do API tokens expire?", "--docs", str(example_docs), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["citations"]


def test_eval_command_writes_reports(example_docs, example_evalset, tmp_path, capsys):
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    code = main([
        "eval", "--docs", str(example_docs), "--evalset", str(example_evalset),
        "--json-out", str(json_out), "--md-out", str(md_out),
    ])
    assert code == 0
    assert "mean" in capsys.readouterr().out
    assert json.loads(json_out.read_text())["num_examples"] == 8
    assert md_out.read_text().startswith("# RAG evaluation report")


def test_eval_quality_gate(example_docs, example_evalset, capsys):
    base = ["eval", "--docs", str(example_docs), "--evalset", str(example_evalset)]
    assert main(base + ["--fail-under", "recall_at_k=0.5"]) == 0
    assert main(base + ["--fail-under", "precision_at_k=1.01"]) == 1
    assert main(base + ["--fail-under", "nonsense=1"]) == 2
    assert "Quality gate failed" in capsys.readouterr().err


def test_missing_docs_is_a_clean_error(tmp_path, capsys):
    assert main(["ingest", "--docs", str(tmp_path / "missing")]) == 2
    assert "rag-lab: error:" in capsys.readouterr().err


def test_python_dash_m_entry_point(example_docs):
    proc = subprocess.run(
        [sys.executable, "-m", "rag_eval_lab", "ingest", "--docs", str(example_docs)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Indexed 5 documents" in proc.stdout
