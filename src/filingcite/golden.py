"""Golden set: questions.yaml (hand written) to golden.jsonl (resolved to accession and page).

python -m filingcite.golden build   resolves every evidence quote against the parsed corpus and fails
                                    when a quote is missing or ambiguous.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import yaml

from . import config
from .ingest import Corpus

GOLDEN_DIR = config.GOLDEN_DIR
QUESTIONS = GOLDEN_DIR / "questions.yaml"
GOLDEN = GOLDEN_DIR / "golden.jsonl"
EVIDENCE_PAGES = GOLDEN_DIR / "evidence_pages.jsonl.gz"
TYPES = ("single_fact", "numeric", "multi_hop", "comparison", "version_trap", "unanswerable")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def find_filing(corpus: Corpus, desc: str) -> str:
    """desc is 'TICKER FORM DATE' where DATE prefixes the period (or the filed date for 8-K)."""
    tk, form, date = desc.split()
    hits = [
        m.accession
        for m in corpus.metas.values()
        if m.ticker == tk and m.form == form and (m.filed if form == "8-K" else m.period).startswith(date)
    ]
    if len(hits) != 1:
        raise ValueError(f"filing '{desc}' matched {len(hits)}: {hits}")
    return hits[0]


def resolve(corpus: Corpus, q: dict) -> dict:
    ev_out = []
    for ev in q.get("evidence") or []:
        acc = find_filing(corpus, ev["filing"])
        want = norm(ev["quote"])
        found = [
            (k, p)
            for k, p in corpus.pages.items()
            if k[0] == acc and want in norm(p.text) and ev.get("doc", k[1]) == k[1] and ev.get("page", k[2]) == k[2]
        ]
        if len(found) != 1:
            raise ValueError(f"{q['id']}: quote matched {len(found)} pages in {ev['filing']}: {want[:80]}")
        (_, doc, pno), pg = found[0]
        seg = (
            next(s for s in pg.segments if want in norm(s.text))
            if any(want in norm(s.text) for s in pg.segments)
            else pg.segments[0]
        )
        ev_out.append(
            {
                "accession": acc,
                "ticker": corpus.metas[acc].ticker,
                "form": corpus.metas[acc].form,
                "document": doc,
                "page": pno,
                "section": seg.section,
                "quote": ev["quote"],
            }
        )
    rec = {k: q[k] for k in ("id", "type", "question") if k in q}
    rec.update(
        {
            "answerable": q["type"] != "unanswerable",
            "answer": q.get("answer", ""),
            "keys": q.get("keys", []),
            "mode": q.get("mode", "latest"),
            "evidence": ev_out,
            "note": q.get("note", ""),
        }
    )
    return rec


def build(corpus: Corpus) -> list[dict]:
    qs = yaml.safe_load(QUESTIONS.read_text())
    seen = set()
    out = []
    errors: list[str] = []
    for q in qs:
        if q["id"] in seen:
            raise ValueError(f"duplicate id {q['id']}")
        seen.add(q["id"])
        if q["type"] not in TYPES:
            raise ValueError(f"{q['id']}: bad type")
        try:
            out.append(resolve(corpus, q))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise ValueError("\n".join(errors))
    GOLDEN.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    needed = {(e["accession"], e["document"], e["page"]) for r in out for e in r["evidence"]}
    with gzip.open(EVIDENCE_PAGES, "wt") as fh:
        for acc, doc, pno in sorted(needed):
            pg = corpus.pages[(acc, doc, pno)]
            m = corpus.metas[acc]
            fh.write(
                json.dumps(
                    {
                        "accession": acc,
                        "ticker": m.ticker,
                        "form": m.form,
                        "period": m.period,
                        "document": doc,
                        "page": pno,
                        "text": pg.text,
                    }
                )
                + "\n"
            )
    return out


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        recs = build(Corpus.load(config.PARSED_DIR))
        from collections import Counter

        print(len(recs), dict(Counter(r["type"] for r in recs)))
