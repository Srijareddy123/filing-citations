"""Raw EDGAR HTML to parsed pages, chunks with metadata, and the current-version flag."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from . import config
from .models import Chunk, FilingMeta, Page
from .parse import parse_document


def fiscal_year(period: str) -> int | None:
    if not period:
        return None
    y, m, d = (int(x) for x in period.split("-"))
    return y - 1 if m == 1 and d <= 7 else y


def build_metas(manifest: list[dict]) -> dict[str, FilingMeta]:
    metas: dict[str, FilingMeta] = {}
    originals = {(r["cik"], r["period"]): r["accession"] for r in manifest if r["form"] == "10-K"}
    for r in manifest:
        first = r["docs"][0]
        is_amend = r["form"].endswith("/A")
        metas[r["accession"]] = FilingMeta(
            accession=r["accession"],
            cik=r["cik"],
            ticker=r["ticker"],
            company=r["company"],
            form=r["form"],
            period=r["period"],
            filed=r["filed"],
            version="amendment" if is_amend else "original",
            amends=originals.get((r["cik"], r["period"])) if is_amend else None,
            fiscal_year=fiscal_year(r["period"]) if r["form"].startswith(("10-K", "10-Q")) else None,
            url=first["url"],
        )
    return metas


def parse_all(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PARSED_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(x) for x in (raw_dir / "manifest.jsonl").read_text().splitlines() if x]
    metas = build_metas(manifest)
    n = 0
    for r in manifest:
        meta = metas[r["accession"]]
        pages: list[Page] = []
        for d in r["docs"]:
            html = (raw_dir / r["ticker"] / r["accession"] / d["filename"]).read_bytes()
            pages += parse_document(html, meta, d["document"])
            for p in pages:
                if p.document == d["document"] and d["document"] != "main":
                    pass
        doc_urls = {d["document"]: d["url"] for d in r["docs"]}
        payload = {"meta": meta.model_dump(), "doc_urls": doc_urls, "pages": [p.model_dump() for p in pages]}
        with gzip.open(out_dir / f"{r['accession']}.json.gz", "wt") as fh:
            json.dump(payload, fh)
        n += 1
    return n


class Corpus:
    """Parsed filings held in memory: metas and pages by accession."""

    def __init__(self) -> None:
        self.metas: dict[str, FilingMeta] = {}
        self.pages: dict[tuple[str, str, int], Page] = {}
        self.doc_urls: dict[str, dict[str, str]] = {}

    @classmethod
    def load(cls, directory: Path) -> Corpus:
        c = cls()
        for f in sorted(directory.glob("*.json.gz")):
            with gzip.open(f, "rt") as fh:
                d = json.load(fh)
            meta = FilingMeta(**d["meta"])
            c.metas[meta.accession] = meta
            c.doc_urls[meta.accession] = d.get("doc_urls", {})
            for p in d["pages"]:
                pg = Page(**p)
                c.pages[(pg.accession, pg.document, pg.page)] = pg
        return c

    def page(self, accession: str, document: str, page: int) -> Page | None:
        return self.pages.get((accession, document, page))

    def page_url(self, accession: str, document: str) -> str:
        urls = self.doc_urls.get(accession, {})
        return urls.get(document) or self.metas[accession].url


def current_map(metas: dict[str, FilingMeta], pages: dict) -> dict[tuple[str, str], bool]:
    """(accession, section) is current unless a later filing of the same cik, period and base form has that section."""
    best: dict[tuple, tuple[str, str]] = {}
    for (acc, doc, _), pg in pages.items():
        if doc != "main":
            continue
        m = metas[acc]
        if not m.form.startswith(("10-K", "10-Q")):
            continue
        for s in pg.segments:
            key = (m.cik, m.period, m.form.split("/")[0], s.section)
            cur = best.get(key)
            if cur is None or (m.filed, acc) > cur:
                best[key] = (m.filed, acc)
    out: dict[tuple[str, str], bool] = {}
    for (acc, doc, _), pg in pages.items():
        m = metas[acc]
        for s in pg.segments:
            if doc == "main" and m.form.startswith(("10-K", "10-Q")):
                key = (m.cik, m.period, m.form.split("/")[0], s.section)
                out[(acc, s.section)] = best[key][1] == acc
            else:
                out[(acc, s.section)] = True
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])")


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for s in _SENT.split(text):
        while len(s) > limit:
            parts.append((cur + " " + s[:limit]).strip() if cur else s[:limit])
            cur, s = "", s[limit:]
        if len(cur) + len(s) + 1 > limit and cur:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def make_chunks(corpus: Corpus, size: int = config.CHUNK_CHARS) -> list[Chunk]:
    cur = current_map(corpus.metas, corpus.pages)
    chunks: list[Chunk] = []
    for (acc, doc, pno), pg in sorted(corpus.pages.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        m = corpus.metas[acc]
        for si, seg in enumerate(pg.segments):
            buf = ""
            pieces: list[str] = []
            for para in seg.text.split("\n"):
                for part in _split_long(para, size):
                    if buf and len(buf) + len(part) + 1 > size:
                        pieces.append(buf)
                        buf = part
                    else:
                        buf = f"{buf}\n{part}" if buf else part
            if buf:
                pieces.append(buf)
            for ci, text in enumerate(pieces):
                if len(text) < 30:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{acc}:{doc}:{pno}:{si}:{ci}",
                        accession=acc,
                        document=doc,
                        page=pno,
                        section=seg.section,
                        section_title=seg.section_title,
                        text=text,
                        cik=m.cik,
                        ticker=m.ticker,
                        company=m.company,
                        form=m.form,
                        period=m.period,
                        filed=m.filed,
                        version=m.version,
                        fiscal_year=m.fiscal_year,
                        current=cur[(acc, seg.section)],
                    )
                )
    return chunks


def embed_text(c: Chunk) -> str:
    return f"{c.company} {c.form} {c.period} {c.section_title}\n{c.text}"


if __name__ == "__main__":
    print(parse_all(), "filings parsed")
