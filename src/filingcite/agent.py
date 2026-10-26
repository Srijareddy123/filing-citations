"""LangGraph agent: retrieve, grade, answer, verify quotes, refuse when support is missing."""

from __future__ import annotations

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rapidfuzz import fuzz

from .ingest import Corpus
from .llm import LLM
from .models import Answer, Citation, Hit, SearchFilters
from .retrieval import Retriever

SYSTEM = (
    "You answer questions about SEC filings using ONLY the numbered passages. "
    'Reply with JSON: {"answerable": true|false, "answer": "short answer", '
    '"citations": [{"source": N, "quote": "exact words copied from passage N"}]}. '
    "Copy quotes character for character. If the passages do not contain the answer, reply "
    '{"answerable": false, "answer": "", "citations": []}. Never use outside knowledge. '
    "If two passages disagree, prefer the one from the later filing."
)
REFUSAL = "The filings in the index do not support an answer to this question."
FUZZY = 88


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def quote_in(quote: str, text: str) -> bool:
    return norm(quote).lower() in norm(text).lower()


class State(TypedDict, total=False):
    question: str
    hits: list[Hit]
    raw: str
    answer: Answer
    ok: bool


def format_passages(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        head = f"{c.company} {c.form} {c.period or c.filed} filed {c.filed} ({c.version}) {c.section_title}, p.{c.page}"
        parts.append(f"[{i}] {head}\n{c.text}")
    return "\n\n".join(parts)


class FilingAgent:
    def __init__(
        self,
        retriever: Retriever,
        corpus: Corpus,
        llm: LLM,
        config_name: str = "hybrid_rerank_version",
        top_k: int = 8,
        min_score: float | None = None,
    ) -> None:
        self.retriever, self.corpus, self.llm = retriever, corpus, llm
        self.config_name, self.top_k, self.min_score = config_name, top_k, min_score
        self.filters: SearchFilters | None = None
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("answer", self._answer)
        g.add_node("verify", self._verify)
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", lambda s: "answer" if s["ok"] else END)
        g.add_edge("answer", "verify")
        g.add_edge("verify", END)
        self.graph = g.compile()

    def _retrieve(self, s: State) -> State:
        hits = self.retriever.search(s["question"], self.config_name, self.top_k, self.filters)
        ok = bool(hits) and (self.min_score is None or hits[0].score >= self.min_score)
        out: State = {"hits": hits, "ok": ok}
        if not ok:
            reason = "no passages matched" if not hits else "top passage scored below the support threshold"
            out["answer"] = Answer(
                question=s["question"],
                answer=REFUSAL,
                answerable=False,
                refusal_reason=reason,
                retrieved=[h.chunk.chunk_id for h in hits],
            )
        return out

    def _answer(self, s: State) -> State:
        user = f"PASSAGES:\n{format_passages(s['hits'])}\n\nQUESTION: {s['question']}"
        return {"raw": self.llm.complete(SYSTEM, user)}

    def _verify(self, s: State) -> State:
        q, hits = s["question"], s["hits"]
        try:
            data = json.loads(s["raw"])
        except (json.JSONDecodeError, TypeError):
            data = {"answerable": False}
        cites: list[Citation] = []
        if data.get("answerable"):
            for c in data.get("citations") or []:
                cit = self._check(c, hits)
                if cit and cit.quote not in {x.quote for x in cites}:
                    cites.append(cit)
        retrieved = [h.chunk.chunk_id for h in hits]
        if data.get("answerable") and cites and str(data.get("answer", "")).strip():
            return {
                "answer": Answer(
                    question=q,
                    answer=str(data["answer"]).strip(),
                    answerable=True,
                    citations=cites,
                    retrieved=retrieved,
                )
            }
        reason = (
            "the model found no support in the passages"
            if not data.get("answerable")
            else "no quote could be verified in the cited page"
        )
        return {
            "answer": Answer(question=q, answer=REFUSAL, answerable=False, refusal_reason=reason, retrieved=retrieved)
        }

    def _check(self, c: dict, hits: list[Hit]) -> Citation | None:
        try:
            src = int(c.get("source"))
            quote = norm(str(c.get("quote", "")))
        except (TypeError, ValueError):
            return None
        if not 1 <= src <= len(hits) or len(quote) < 8:
            return None
        h = hits[src - 1].chunk
        page = self.corpus.page(h.accession, h.document, h.page)
        page_text = page.text if page else h.text
        if not quote_in(quote, page_text):
            snapped = self._snap(quote, h.text)
            if not snapped:
                return None
            quote = snapped
        m = self.corpus.metas[h.accession]
        return Citation(
            accession=h.accession,
            company=h.company,
            ticker=h.ticker,
            form=h.form,
            period=h.period,
            filed=h.filed,
            version=h.version,
            document=h.document,
            section=h.section,
            section_title=h.section_title,
            page=h.page,
            quote=quote,
            verified=True,
            url=self.corpus.page_url(h.accession, h.document) if m else "",
        )

    @staticmethod
    def _snap(quote: str, text: str) -> str | None:
        """Replace a near-verbatim quote with the closest sentence that really is in the text."""
        best, score = None, 0.0
        for sent in re.split(r"(?<=[.;:])\s+|\n", text):
            if len(sent) < 8:
                continue
            r = fuzz.partial_ratio(quote.lower(), sent.lower())
            if r > score:
                best, score = sent.strip(), r
        return best if best and score >= FUZZY and len(best) <= 600 else None

    def ask(self, question: str, filters: SearchFilters | None = None) -> Answer:
        self.filters = filters
        t0 = time.perf_counter()
        out = self.graph.invoke({"question": question})
        ans: Answer = out["answer"]
        ans.model = self.llm.name
        ans.latency_s = time.perf_counter() - t0
        return ans
