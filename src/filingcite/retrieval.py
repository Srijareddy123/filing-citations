"""Hybrid retrieval: BM25 + dense, reciprocal rank fusion, cross-encoder rerank, scope and version filters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .backends import Backend
from .embed import get_embedder, get_reranker
from .models import FilingMeta, Hit, SearchFilters

RRF_K = 60
CANDIDATES = 50
RERANK_POOL = 30

CONFIGS: dict[str, dict] = {
    "bm25": {"mode": "bm25", "rerank": False, "current": False},
    "dense": {"mode": "dense", "rerank": False, "current": False},
    "hybrid": {"mode": "hybrid", "rerank": False, "current": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank": True, "current": False},
    "hybrid_rerank_version": {"mode": "hybrid", "rerank": True, "current": True},
}
COMMON_FIRST_WORDS = {"home", "walt", "advanced", "general", "united", "american", "coca", "first", "new"}
_ORIGINAL = re.compile(
    r"\b(original(ly)?|as (first )?filed|before the amendment|prior to the amendment|"
    r"superseded)\b",
    re.I,
)
_FORM = re.compile(r"\b(10-K/A|10-K|10-Q|8-K)\b", re.I)
_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|co|company|ltd|plc|holdings|group|technologies|platforms|"
    r"communications|the|de|new|inc\.|/new)\b\.?",
    re.I,
)


def build_aliases(metas: dict[str, FilingMeta]) -> dict[str, str]:
    """Lowercase company-name alias to ticker."""
    out: dict[str, str] = {}
    for m in metas.values():
        name = re.sub(r"[^a-z0-9 ]", " ", _SUFFIX.sub(" ", m.company.lower()))
        name = " ".join(name.split())
        if name:
            out[name] = m.ticker
            first = name.split()[0]
            if len(first) >= 4 and first not in COMMON_FIRST_WORDS:
                out[first] = m.ticker
    out.update(
        {
            "alphabet": "GOOGL",
            "google": "GOOGL",
            "facebook": "META",
            "ford": "F",
            "mrk": "MRK",
            "merck": "MRK",
            "disney": "DIS",
            "walt disney": "DIS",
            "doordash": "DASH",
            "ebay": "EBAY",
            "coca cola": "KO",
            "coca-cola": "KO",
            "advanced micro devices": "AMD",
            "boeing": "BA",
            "caterpillar": "CAT",
            "home depot": "HD",
            "walmart": "WMT",
        }
    )
    return {
        k: v
        for k, v in out.items()
        if k in {m.company.lower() for m in metas.values()} or v in {m.ticker for m in metas.values()}
    }


def parse_scope(question: str, aliases: dict[str, str], current: bool) -> SearchFilters:
    """Company and form named in the question become filters. The version filter is switched by config."""
    q = question.lower()
    tickers = sorted({tk for name, tk in aliases.items() if re.search(rf"\b{re.escape(name)}\b", q)})
    for tk in set(aliases.values()):
        if re.search(rf"\b{re.escape(tk)}\b", question) and len(tk) > 1:
            tickers = sorted(set(tickers) | {tk})
    forms = sorted({f.upper() for f in _FORM.findall(question)})
    if "10-K" in forms and "10-K/A" not in forms:
        forms.append("10-K/A")  # a question about "the 10-K" may be answered by its amendment
    f = SearchFilters(tickers=tickers, forms=forms)
    if current:
        if _ORIGINAL.search(question):
            f.version = "original"
        else:
            f.current_only = True
    return f


def rrf(lists: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    score: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.chunk_id
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            keep.setdefault(cid, h)
    order = sorted(score, key=lambda c: -score[c])
    return [Hit(chunk=keep[c].chunk, score=score[c], source="rrf") for c in order]


@dataclass
class Retriever:
    backend: Backend
    aliases: dict[str, str]
    embed_model: str | None = None
    rerank_model: str | None = None

    def search(
        self,
        question: str,
        config_name: str = "hybrid_rerank_version",
        top_k: int = 8,
        filters: SearchFilters | None = None,
        use_scope: bool = True,
    ) -> list[Hit]:
        cfg = CONFIGS[config_name]
        f = filters or (parse_scope(question, self.aliases, cfg["current"]) if use_scope else SearchFilters())
        if filters is None and not use_scope and cfg["current"]:
            f.current_only = True
        mode = cfg["mode"]
        lists: list[list[Hit]] = []
        if mode in ("bm25", "hybrid"):
            lists.append(self.backend.bm25(question, f, CANDIDATES))
        if mode in ("dense", "hybrid"):
            qv = get_embedder(self.embed_model).encode([question], query=True)[0]
            lists.append(self.backend.dense(qv, f, CANDIDATES))
        hits = lists[0] if len(lists) == 1 else rrf(lists)
        if cfg["rerank"] and hits:
            pool = hits[:RERANK_POOL]
            scores = get_reranker(self.rerank_model).score(question, [h.chunk.text for h in pool])
            ranked = sorted(zip(pool, scores, strict=True), key=lambda x: -x[1])
            return [Hit(chunk=h.chunk, score=s, source="rerank") for h, s in ranked[:top_k]]
        return hits[:top_k]


def default_retriever(corpus_dir=None) -> Retriever:
    """OpenSearch if FC_BACKEND=opensearch, else build the in-memory backend from parsed files."""
    from .ingest import Corpus, make_chunks

    corpus = Corpus.load(corpus_dir or config.PARSED_DIR)
    aliases = build_aliases(corpus.metas)
    if config.BACKEND == "opensearch":
        from .backends import OpenSearchBackend

        return Retriever(OpenSearchBackend(), aliases)
    from .backends import MemoryBackend
    from .ingest import embed_text

    chunks = make_chunks(corpus)
    vecs = get_embedder().encode([embed_text(c) for c in chunks])
    return Retriever(MemoryBackend(chunks, vecs), aliases)
