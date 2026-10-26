"""Pydantic models shared by ingest, search, agent, API and MCP."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Version = Literal["original", "amendment"]


class FilingMeta(BaseModel):
    accession: str
    cik: int
    ticker: str
    company: str
    form: str  # 10-K, 10-K/A, 10-Q, 8-K
    period: str  # period of report, YYYY-MM-DD ("" for some 8-Ks)
    filed: str  # YYYY-MM-DD
    version: Version = "original"
    amends: str | None = None  # accession of the filing this one amends
    fiscal_year: int | None = None
    url: str = ""


class Segment(BaseModel):
    section: str  # normalized key, e.g. "item_1a", "cover", "exhibit"
    section_title: str
    text: str


class Page(BaseModel):
    accession: str
    document: str  # "main" or an exhibit type such as "EX-10.1"
    page: int  # 1-based position of the page break in the filing HTML
    segments: list[Segment]
    page_source: Literal["break", "synthetic"] = "break"

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)


class Chunk(BaseModel):
    chunk_id: str
    accession: str
    document: str
    page: int
    section: str
    section_title: str
    text: str
    cik: int
    ticker: str
    company: str
    form: str
    period: str
    filed: str
    version: Version
    fiscal_year: int | None = None
    current: bool = True  # False when a later amendment replaced this section


class Citation(BaseModel):
    accession: str
    company: str
    ticker: str
    form: str
    period: str
    filed: str
    version: Version
    document: str
    section: str
    section_title: str = ""
    page: int
    quote: str
    verified: bool = False  # quote found verbatim in the cited page
    url: str = ""


class Answer(BaseModel):
    question: str
    answer: str
    answerable: bool
    citations: list[Citation] = Field(default_factory=list)
    refusal_reason: str = ""
    model: str = ""
    latency_s: float = 0.0
    retrieved: list[str] = Field(default_factory=list)  # chunk ids, for tracing


class SearchFilters(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    forms: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    filed_after: str | None = None
    filed_before: str | None = None
    current_only: bool = False
    version: Version | None = None


class Hit(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # which retriever / stage produced the score
