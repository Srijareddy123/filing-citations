from fastapi.testclient import TestClient

from filingcite.api import create_app, highlight
from filingcite.llm import ExtractiveLLM


def client(services):
    return TestClient(create_app(services, ExtractiveLLM()))


def test_highlight_wraps_quote_across_whitespace():
    out = highlight("a  b <c>\nd e f", "b <c> d")
    assert '<mark id="quote">b &lt;c&gt;\nd</mark>' in out


def test_ask_returns_cited_answer(services):
    r = client(services).post(
        "/api/ask",
        json={
            "question": "What date does Merck's current 10-K for fiscal 2023 "
            "give for its Annual Meeting of Shareholders?"
        },
    )
    body = r.json()
    assert r.status_code == 200 and body["answerable"] and body["citations"][0]["verified"]


def test_bad_config_is_rejected(services):
    r = client(services).post("/api/ask", json={"question": "x", "config": "nope"})
    assert r.status_code == 400


def test_view_marks_the_quote(services):
    c = client(services)
    body = c.post(
        "/api/ask",
        json={
            "question": "What date does Merck's current 10-K for fiscal 2023 give for its "
            "Annual Meeting of Shareholders?"
        },
    ).json()
    cit = body["citations"][0]
    r = c.get(
        "/view",
        params={
            "accession": cit["accession"],
            "document": cit["document"],
            "page": cit["page"],
            "quote": cit["quote"],
        },
    )
    assert r.status_code == 200 and '<mark id="quote">' in r.text


def test_view_404_and_index(services):
    c = client(services)
    assert c.get("/view", params={"accession": "nope", "page": 1}).status_code == 404
    assert "Filing" in c.get("/").text
