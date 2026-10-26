from filingcite.ingest import current_map, fiscal_year


def test_fiscal_year_for_early_january_year_end():
    assert fiscal_year("2026-01-03") == 2025
    assert fiscal_year("2025-12-27") == 2025
    assert fiscal_year("2026-01-31") == 2026


def test_amended_sections_supersede_the_original(corpus):
    cur = current_map(corpus.metas, corpus.pages)
    orig = next(a for a, m in corpus.metas.items() if m.ticker == "MRK" and m.form == "10-K")
    amend = next(a for a, m in corpus.metas.items() if m.ticker == "MRK" and m.form == "10-K/A")
    assert cur[(amend, "cover")] is True
    assert cur[(orig, "cover")] is False


def test_amendment_links_to_original(corpus):
    for m in corpus.metas.values():
        if m.form == "10-K/A" and m.ticker in {"MRK", "AMD"}:
            assert m.version == "amendment"
            assert m.amends in corpus.metas


def test_chunks_carry_metadata_and_size(chunks):
    assert chunks
    assert all(c.text and c.ticker and c.accession for c in chunks)
    assert max(len(c.text) for c in chunks) < 2600
    assert {c.version for c in chunks} == {"original", "amendment"}
