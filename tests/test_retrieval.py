from filingcite.models import Hit, SearchFilters
from filingcite.retrieval import CONFIGS, parse_scope, rrf


def test_scope_parser_finds_company_form_and_original(retriever):
    f = parse_scope("What did Merck's original 10-K say?", retriever.aliases, current=True)
    assert f.tickers == ["MRK"] and f.forms == ["10-K", "10-K/A"] and f.version == "original"
    f2 = parse_scope("What did Merck's 10-K say?", retriever.aliases, current=True)
    assert f2.current_only and f2.version is None
    f3 = parse_scope("What did Merck's 10-K say?", retriever.aliases, current=False)
    assert not f3.current_only


def test_rrf_prefers_items_ranked_high_in_both_lists(chunks):
    a = [Hit(chunk=chunks[i], score=1) for i in (0, 1, 2)]
    b = [Hit(chunk=chunks[i], score=1) for i in (2, 1, 3)]
    assert [h.chunk.chunk_id for h in rrf([a, b])][:2] == [chunks[2].chunk_id, chunks[1].chunk_id]


def test_all_configs_return_hits(retriever):
    for name in CONFIGS:
        hits = retriever.search("Merck annual meeting of shareholders date", name, 5)
        assert hits and all(h.chunk.ticker == "MRK" for h in hits)


def test_version_filter_never_returns_superseded_text(retriever):
    q = "Merck annual meeting of shareholders May 2024"
    stale = [h for h in retriever.search(q, "hybrid", 8) if not h.chunk.current]
    assert stale, "the unfiltered search should hit the superseded original"
    filtered = retriever.search(q, "hybrid_rerank_version", 8)
    assert filtered and all(h.chunk.current for h in filtered)


def test_explicit_filters(retriever):
    f = SearchFilters(tickers=["AMD"], forms=["10-K/A"], current_only=True)
    hits = retriever.search("client net revenue unit shipments", "hybrid", 5, filters=f)
    assert hits and all(h.chunk.form == "10-K/A" for h in hits)
    late = SearchFilters(tickers=["AMD"], filed_after="2099-01-01")
    assert retriever.search("client net revenue", "bm25", 5, filters=late) == []
