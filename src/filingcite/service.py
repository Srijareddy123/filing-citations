"""Functions shared by the web API and the MCP server."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .ingest import Corpus
from .models import SearchFilters
from .retrieval import Retriever


@dataclass
class Services:
    retriever: Retriever
    corpus: Corpus

    def search_filings(
        self,
        query: str,
        ticker: str | None = None,
        form: str | None = None,
        fiscal_year: int | None = None,
        filed_after: str | None = None,
        filed_before: str | None = None,
        current_only: bool = True,
        top_k: int = 5,
        rerank: bool = True,
    ) -> list[dict]:
        f = SearchFilters(
            tickers=[ticker.upper()] if ticker else [],
            forms=[form.upper()] if form else [],
            fiscal_years=[fiscal_year] if fiscal_year else [],
            filed_after=filed_after,
            filed_before=filed_before,
            current_only=current_only,
        )
        cfg = "hybrid_rerank" if rerank else "hybrid"
        out = []
        for h in self.retriever.search(query, cfg, top_k, filters=f):
            c = h.chunk
            out.append(
                {
                    "accession": c.accession,
                    "ticker": c.ticker,
                    "company": c.company,
                    "form": c.form,
                    "period": c.period,
                    "filed": c.filed,
                    "version": c.version,
                    "current": c.current,
                    "document": c.document,
                    "section": c.section,
                    "section_title": c.section_title,
                    "page": c.page,
                    "score": round(h.score, 4),
                    "text": c.text,
                }
            )
        return out

    def get_section(self, accession: str, section: str, document: str = "main") -> dict:
        pages = sorted(
            (p for (a, d, _), p in self.corpus.pages.items() if a == accession and d == document),
            key=lambda p: p.page,
        )
        parts = []
        for p in pages:
            for s in p.segments:
                if s.section == section:
                    parts.append({"page": p.page, "title": s.section_title, "text": s.text})
        if not parts:
            return {"error": f"no section '{section}' in {accession} ({document})"}
        m = self.corpus.metas[accession]
        return {
            "accession": accession,
            "company": m.company,
            "form": m.form,
            "period": m.period,
            "filed": m.filed,
            "version": m.version,
            "section": section,
            "pages": parts,
        }

    def _section_text(self, accession: str, section: str) -> str:
        res = self.get_section(accession, section)
        return "\n".join(p["text"] for p in res.get("pages", []))

    def compare_versions(self, ticker: str, period: str, section: str | None = None, max_diff_lines: int = 60) -> dict:
        """List the original and amended 10-K filings for a period; diff one section between them."""
        fl = sorted(
            (
                m
                for m in self.corpus.metas.values()
                if m.ticker == ticker.upper() and m.form.startswith("10-K") and m.period.startswith(period)
            ),
            key=lambda m: m.filed,
        )
        if len(fl) < 2:
            return {"error": f"need an original and an amendment for {ticker} {period}; found {len(fl)}"}
        sections: dict[str, list[str]] = {}
        for m in fl:
            keys = {
                s.section
                for (a, d, _), p in self.corpus.pages.items()
                if a == m.accession and d == "main"
                for s in p.segments
            }
            sections[m.accession] = sorted(keys)
        res: dict = {
            "filings": [
                {
                    "accession": m.accession,
                    "form": m.form,
                    "filed": m.filed,
                    "version": m.version,
                    "sections": sections[m.accession],
                }
                for m in fl
            ]
        }
        if section:
            old, new = fl[0].accession, fl[-1].accession
            a, b = self._section_text(old, section), self._section_text(new, section)
            if not a or not b:
                res["diff"] = f"section '{section}' is missing from {'the original' if not a else 'the amendment'}"
                return res
            sa, sb = re.split(r"(?<=[.;])\s+|\n", a), re.split(r"(?<=[.;])\s+|\n", b)
            diff = [
                x
                for x in difflib.unified_diff(sa, sb, "original", "amendment", lineterm="", n=0)
                if not x.startswith(("---", "+++", "@@"))
            ]
            res["diff"] = diff[:max_diff_lines]
            res["changed_lines"] = len(diff)
        return res
