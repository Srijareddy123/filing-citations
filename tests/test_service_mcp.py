import asyncio

from filingcite import mcp_server


def test_search_filings_filters_and_fields(services):
    res = services.search_filings("annual meeting date", ticker="MRK", form="10-K/A", top_k=3)
    assert res and all(r["ticker"] == "MRK" and r["form"] == "10-K/A" for r in res)
    assert {"accession", "section", "page", "version", "current", "text"} <= set(res[0])


def test_get_section_returns_pages(services, corpus):
    acc = next(a for a, m in corpus.metas.items() if m.ticker == "MRK" and m.form == "10-K/A")
    res = services.get_section(acc, "item_10")
    assert res["pages"] and res["version"] == "amendment"
    assert "error" in services.get_section(acc, "item_99")


def test_compare_versions_diffs_amd_md_and_a(services):
    res = services.compare_versions("AMD", "2025", "item_7")
    forms = [f["form"] for f in res["filings"]]
    assert forms == ["10-K", "10-K/A"]
    text = "\n".join(res["diff"])
    assert "31% increase in unit shipments" in text and "15% increase in unit shipments" in text


def test_compare_versions_needs_two_filings(services):
    assert "error" in services.compare_versions("UBER", "2025")


def test_mcp_server_exposes_three_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in (tools.tools if hasattr(tools, "tools") else tools)}
    assert names == {"search_filings", "get_section", "compare_versions"}
