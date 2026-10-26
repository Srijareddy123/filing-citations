import os

# Offline stand-ins so the suite needs no Docker, no model downloads and no API keys.
os.environ["FC_BACKEND"] = "memory"
os.environ["FC_EMBED_MODEL"] = "hash"
os.environ["FC_RERANK_MODEL"] = "lexical"
os.environ["FC_LLM"] = "extractive"

import pytest  # noqa: E402

from filingcite import config  # noqa: E402
from filingcite.backends import MemoryBackend  # noqa: E402
from filingcite.embed import get_embedder  # noqa: E402
from filingcite.ingest import Corpus, embed_text, make_chunks  # noqa: E402
from filingcite.retrieval import Retriever, build_aliases  # noqa: E402
from filingcite.service import Services  # noqa: E402


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.load(config.SAMPLE_DIR)


@pytest.fixture(scope="session")
def chunks(corpus):
    return make_chunks(corpus)


@pytest.fixture(scope="session")
def retriever(corpus, chunks) -> Retriever:
    vecs = get_embedder("hash").encode([embed_text(c) for c in chunks])
    return Retriever(
        MemoryBackend(chunks, vecs), build_aliases(corpus.metas), embed_model="hash", rerank_model="lexical"
    )


@pytest.fixture(scope="session")
def services(retriever, corpus) -> Services:
    return Services(retriever, corpus)
