import gzip
import json
from collections import Counter

import pytest

from filingcite import config
from filingcite.golden import EVIDENCE_PAGES, load_golden, norm


@pytest.fixture(scope="module")
def golden():
    return load_golden()


@pytest.fixture(scope="module")
def evidence_pages():
    with gzip.open(EVIDENCE_PAGES, "rt") as fh:
        rows = [json.loads(x) for x in fh]
    return {(r["accession"], r["document"], r["page"]): r["text"] for r in rows}


def test_size_and_mix(golden):
    counts = Counter(r["type"] for r in golden)
    assert len(golden) >= 120
    assert counts["unanswerable"] >= 30
    for t in ("single_fact", "multi_hop", "comparison", "numeric", "version_trap"):
        assert counts[t] >= 10, t
    assert len({r["id"] for r in golden}) == len(golden)


def test_every_answerable_question_has_keys_and_evidence(golden):
    for r in golden:
        if r["answerable"]:
            assert r["evidence"] and r["keys"] and r["answer"], r["id"]
        else:
            assert not r["evidence"] and r["note"], r["id"]
    assert all(len(r["evidence"]) >= 2 for r in golden if r["type"] in ("comparison",))


def test_every_evidence_quote_is_verbatim_in_the_indexed_page(golden, evidence_pages):
    checked = 0
    for r in golden:
        for e in r["evidence"]:
            text = evidence_pages[(e["accession"], e["document"], e["page"])]
            assert norm(e["quote"]) in norm(text), f"{r['id']}: quote not in page"
            checked += 1
    assert checked >= 100


def test_quotes_match_the_full_parsed_corpus_when_present(golden, evidence_pages):
    from filingcite.ingest import Corpus

    if not any(config.PARSED_DIR.glob("*.json.gz")):
        pytest.skip("full corpus not fetched")
    corpus = Corpus.load(config.PARSED_DIR)
    for key, text in evidence_pages.items():
        assert norm(corpus.pages[key].text) == norm(text), key
