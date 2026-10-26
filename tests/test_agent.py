import json

from filingcite.agent import REFUSAL, FilingAgent, quote_in
from filingcite.llm import ExtractiveLLM


class Canned:
    name = "canned"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(self.payload) if not isinstance(self.payload, str) else self.payload


def agent(retriever, corpus, llm, cfg="hybrid_rerank_version"):
    return FilingAgent(retriever, corpus, llm, config_name=cfg)


def test_answer_has_verified_quotable_citation(retriever, corpus):
    a = agent(retriever, corpus, ExtractiveLLM()).ask(
        "What date does Merck's current 10-K for fiscal 2023 give for its Annual Meeting of Shareholders?"
    )
    assert a.answerable and a.citations
    c = a.citations[0]
    page = corpus.page(c.accession, c.document, c.page)
    assert c.verified and quote_in(c.quote, page.text)
    assert "May 28, 2024" in a.answer


def test_refuses_when_nothing_supports_an_answer(retriever, corpus):
    a = agent(retriever, corpus, ExtractiveLLM()).ask("What was JPMorgan Chase's net interest income in 2025?")
    assert not a.answerable and a.answer == REFUSAL and not a.citations


def test_invented_quote_is_dropped_and_answer_refused(retriever, corpus):
    llm = Canned(
        {
            "answerable": True,
            "answer": "It was 42.",
            "citations": [{"source": 1, "quote": "This sentence does not appear anywhere in the filing at all."}],
        }
    )
    a = agent(retriever, corpus, llm).ask("Merck annual meeting date")
    assert not a.answerable and "no quote could be verified" in a.refusal_reason


def test_near_verbatim_quote_snaps_to_real_text(retriever, corpus):
    hits = retriever.search("Merck annual meeting", "hybrid_rerank_version", 6)
    real = hits[0].chunk.text.split("\n")[0]
    sloppy = real[:-3] + "xyz"
    llm = Canned({"answerable": True, "answer": "ok", "citations": [{"source": 1, "quote": sloppy}]})
    a = agent(retriever, corpus, llm).ask("Merck annual meeting")
    if a.answerable:
        assert quote_in(a.citations[0].quote, hits[0].chunk.text)


def test_bad_json_means_refusal(retriever, corpus):
    a = agent(retriever, corpus, Canned("not json")).ask("Merck annual meeting")
    assert not a.answerable
