"""MCP server: search_filings, get_section, compare_versions. Run: python -m filingcite.mcp_server"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import config
from .ingest import Corpus
from .retrieval import default_retriever

mcp = MCPServer(
    "filing-citations",
    instructions=(
        "Search SEC 10-K, 10-Q and 8-K filings. Every result carries accession, section and page so answers can "
        "cite them. By default only the current version of each section is searched."
    ),
)
_svc = None


def svc():
    global _svc
    if _svc is None:
        from .service import Services

        _svc = Services(default_retriever(), Corpus.load(config.PARSED_DIR))
    return _svc


@mcp.tool()
def search_filings(
    query: str,
    ticker: str | None = None,
    form: str | None = None,
    fiscal_year: int | None = None,
    filed_after: str | None = None,
    filed_before: str | None = None,
    current_only: bool = True,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid search (BM25 + dense, reranked) with company, form, fiscal year and date filters."""
    return svc().search_filings(query, ticker, form, fiscal_year, filed_after, filed_before, current_only, top_k)


@mcp.tool()
def get_section(accession: str, section: str, document: str = "main") -> dict:
    """Full text of one section (for example item_1a or item_7) of a filing, with page numbers."""
    return svc().get_section(accession, section, document)


@mcp.tool()
def compare_versions(ticker: str, period: str, section: str | None = None) -> dict:
    """List the original and amended 10-K for a period and diff one section between them."""
    return svc().compare_versions(ticker, period, section)


if __name__ == "__main__":
    mcp.run()
