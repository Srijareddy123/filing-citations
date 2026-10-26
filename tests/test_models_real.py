import pytest

from filingcite.backends import MemoryBackend
from filingcite.embed import get_embedder, get_reranker
from filingcite.ingest import embed_text
from filingcite.retrieval import Retriever, build_aliases


@pytest.mark.slow
def test_real_models_rank_the_right_passage(corpus, chunks):
    emb = get_embedder("BAAI/bge-small-en-v1.5")
    vecs = emb.encode([embed_text(c) for c in chunks])
    r = Retriever(
        MemoryBackend(chunks, vecs),
        build_aliases(corpus.metas),
        embed_model="BAAI/bge-small-en-v1.5",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    hits = r.search("What cash price per Delivery Hero share is Uber offering?", "hybrid_rerank", 3)
    assert "41.50" in " ".join(h.chunk.text for h in hits)
    assert (
        get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2").score("cat", ["a cat sat", "stock markets"])[0]
        > get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2").score("cat", ["a cat sat", "stock markets"])[1]
    )
