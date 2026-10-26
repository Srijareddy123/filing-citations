import httpx
import pytest

from filingcite import config
from filingcite.backends import OpenSearchBackend
from filingcite.embed import get_embedder
from filingcite.ingest import embed_text
from filingcite.models import SearchFilters


def reachable() -> bool:
    try:
        return httpx.get(config.OPENSEARCH_URL, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not reachable(), reason="OpenSearch not running on OPENSEARCH_URL")
def test_opensearch_bm25_dense_and_filters(chunks):
    emb = get_embedder("hash")
    be = OpenSearchBackend(index="filings_test")
    be.create_index(emb.dim)
    be.add(chunks, emb.encode([embed_text(c) for c in chunks]))
    f = SearchFilters(tickers=["MRK"], current_only=True)
    bm = be.bm25("annual meeting of shareholders", f, 5)
    assert bm and all(h.chunk.ticker == "MRK" and h.chunk.current for h in bm)
    dn = be.dense(emb.encode(["annual meeting of shareholders"], query=True)[0], f, 5)
    assert dn and all(h.chunk.current for h in dn)
    assert be.get_chunks(bm[0].chunk.accession, bm[0].chunk.section)
    be.es.indices.delete(index="filings_test")
