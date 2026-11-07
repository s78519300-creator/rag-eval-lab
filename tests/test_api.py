from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from rag_eval_lab.api import create_server, handle_ask


@pytest.fixture
def server_url(pipeline):
    server = create_server(pipeline, port=0, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _post(url, body: bytes):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_handle_ask_validation(pipeline):
    assert handle_ask(pipeline, [])[0] == 400
    assert handle_ask(pipeline, {"question": "  "})[0] == 400
    assert handle_ask(pipeline, {"question": "refund", "top_k": 0})[0] == 400
    status, body = handle_ask(pipeline, {"question": "When do API tokens expire?", "top_k": 2})
    assert status == 200
    assert len(body["retrieved"]) <= 2


def test_post_ask_over_http(server_url):
    payload = json.dumps({"question": "How is customer data encrypted at rest?"}).encode()
    status, body = _post(f"{server_url}/ask", payload)
    assert status == 200
    assert "AES-256" in body["answer"]
    assert body["citations"][0].startswith("security.md#chunk")


def test_http_errors(server_url):
    assert _post(f"{server_url}/ask", b"{oops")[0] == 400
    assert _post(f"{server_url}/nope", b"{}")[0] == 404
    with urllib.request.urlopen(f"{server_url}/health", timeout=5) as response:
        assert json.loads(response.read())["status"] == "ok"
