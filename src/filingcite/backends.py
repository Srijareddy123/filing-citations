"""Retrieval backends: OpenSearch (full path) and an in-memory twin (tests, no Docker)."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .models import Chunk, Hit, SearchFilters

_TOK = re.compile(r"[a-z0-9]+")


class Backend(Protocol):
    def bm25(self, query: str, f: SearchFilters, k: int) -> list[Hit]: ...
    def dense(self, qvec: np.ndarray, f: SearchFilters, k: int) -> list[Hit]: ...
    def get_chunks(self, accession: str, section: str | None = None) -> list[Chunk]: ...


def matches(c: Chunk, f: SearchFilters) -> bool:
    if f.tickers and c.ticker not in f.tickers:
        return False
    if f.forms and c.form not in f.forms:
        return False
    if f.fiscal_years and c.fiscal_year not in f.fiscal_years:
        return False
    if f.filed_after and c.filed < f.filed_after:
        return False
    if f.filed_before and c.filed > f.filed_before:
        return False
    if f.current_only and not c.current:
        return False
    return not (f.version and c.version != f.version)


def os_filter(f: SearchFilters) -> list[dict]:
    out: list[dict] = []
    if f.tickers:
        out.append({"terms": {"ticker": f.tickers}})
    if f.forms:
        out.append({"terms": {"form": f.forms}})
    if f.fiscal_years:
        out.append({"terms": {"fiscal_year": f.fiscal_years}})
    if f.filed_after or f.filed_before:
        out.append({"range": {"filed": {k: v for k, v in (("gte", f.filed_after), ("lte", f.filed_before)) if v}}})
    if f.current_only:
        out.append({"term": {"current": True}})
    if f.version:
        out.append({"term": {"version": f.version}})
    return out


class MemoryBackend:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.bm = BM25Okapi([_TOK.findall(c.text.lower()) for c in chunks])

    def bm25(self, query: str, f: SearchFilters, k: int) -> list[Hit]:
        scores = self.bm.get_scores(_TOK.findall(query.lower()))
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            if matches(self.chunks[i], f):
                hits.append(Hit(chunk=self.chunks[i], score=float(scores[i]), source="bm25"))
        return hits

    def dense(self, qvec: np.ndarray, f: SearchFilters, k: int) -> list[Hit]:
        sims = self.vectors @ qvec
        order = np.argsort(-sims)
        hits = []
        for i in order:
            if len(hits) >= k:
                break
            if matches(self.chunks[i], f):
                hits.append(Hit(chunk=self.chunks[i], score=float(sims[i]), source="dense"))
        return hits

    def get_chunks(self, accession: str, section: str | None = None) -> list[Chunk]:
        return [c for c in self.chunks if c.accession == accession and (section is None or c.section == section)]


class OpenSearchBackend:
    def __init__(self, url: str | None = None, index: str | None = None) -> None:
        from opensearchpy import OpenSearch

        self.es = OpenSearch(url or config.OPENSEARCH_URL, timeout=60)
        self.index = index or config.INDEX_NAME

    def create_index(self, dim: int) -> None:
        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)
        kw = {"type": "keyword"}
        self.es.indices.create(
            index=self.index,
            body={
                "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "section_title": {"type": "text"},
                        "chunk_id": kw,
                        "accession": kw,
                        "document": kw,
                        "section": kw,
                        "ticker": kw,
                        "form": kw,
                        "company": kw,
                        "version": kw,
                        "period": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                        "filed": {"type": "date", "format": "yyyy-MM-dd"},
                        "fiscal_year": {"type": "integer"},
                        "cik": {"type": "long"},
                        "page": {"type": "integer"},
                        "current": {"type": "boolean"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dim,
                            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                        },
                    }
                },
            },
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 500) -> None:
        from opensearchpy import helpers

        for i in range(0, len(chunks), batch):
            actions = (
                {
                    "_index": self.index,
                    "_id": c.chunk_id,
                    "_source": {**c.model_dump(), "period": c.period or None, "embedding": v.tolist()},
                }
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
            )
            helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index)

    @staticmethod
    def _to_chunk(src: dict) -> Chunk:
        src = {k: v for k, v in src.items() if k != "embedding"}
        src["period"] = (src.get("period") or "")[:10]
        return Chunk(**src)

    def bm25(self, query: str, f: SearchFilters, k: int) -> list[Hit]:
        body = {
            "size": k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "bool": {
                    "must": [{"multi_match": {"query": query, "fields": ["text", "section_title^0.5"]}}],
                    "filter": os_filter(f),
                }
            },
        }
        r = self.es.search(index=self.index, body=body)
        return [Hit(chunk=self._to_chunk(h["_source"]), score=h["_score"], source="bm25") for h in r["hits"]["hits"]]

    def dense(self, qvec: np.ndarray, f: SearchFilters, k: int) -> list[Hit]:
        knn: dict = {"vector": qvec.tolist(), "k": k}
        flt = os_filter(f)
        if flt:
            knn["filter"] = {"bool": {"filter": flt}}
        body = {"size": k, "_source": {"excludes": ["embedding"]}, "query": {"knn": {"embedding": knn}}}
        r = self.es.search(index=self.index, body=body)
        return [Hit(chunk=self._to_chunk(h["_source"]), score=h["_score"], source="dense") for h in r["hits"]["hits"]]

    def get_chunks(self, accession: str, section: str | None = None) -> list[Chunk]:
        flt: list[dict] = [{"term": {"accession": accession}}]
        if section:
            flt.append({"term": {"section": section}})
        r = self.es.search(
            index=self.index,
            body={
                "size": 1000,
                "_source": {"excludes": ["embedding"]},
                "query": {"bool": {"filter": flt}},
                "sort": [{"page": "asc"}, {"chunk_id": "asc"}],
            },
        )
        return [self._to_chunk(h["_source"]) for h in r["hits"]["hits"]]
