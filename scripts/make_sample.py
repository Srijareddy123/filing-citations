"""Build data/sample from the full parsed corpus: a few small filings used by `make demo` and the tests.

Full filings are kept whole when small. Large originals keep only the cover and the pages the golden set cites,
so the amendment pairs (original vs amended text) stay intact.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys

from filingcite import config
from filingcite.golden import find_filing, load_golden
from filingcite.ingest import Corpus

FULL = [
    "AMD 10-K 2025",
    "AMD 10-K/A 2025",
    "EBAY 10-K/A 2024",
    "MRK 10-K/A 2023",
    "DIS 10-K/A 2023-09",
    "UBER 8-K 2026-07-16",
    "BA 8-K 2026-08-28",
    "TSLA 8-K 2026-09-29",
    "NVDA 8-K 2026-08-17",
]
EXCERPT = ["MRK 10-K 2023", "EBAY 10-K 2024", "DIS 10-K 2023-09"]
MAIN_ONLY = {"UBER 8-K 2026-07-16", "BA 8-K 2026-08-28", "TSLA 8-K 2026-09-29", "NVDA 8-K 2026-08-17"}


def main() -> None:
    corpus = Corpus.load(config.PARSED_DIR)
    out = config.SAMPLE_DIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cited = {(e["accession"], e["page"]) for r in load_golden() for e in r["evidence"]}
    for desc, full in [(d, True) for d in FULL] + [(d, False) for d in EXCERPT]:
        acc = find_filing(corpus, desc)
        meta = corpus.metas[acc]
        pages = []
        for (a, doc, pno), pg in sorted(corpus.pages.items()):
            if a != acc or (desc in MAIN_ONLY and doc != "main"):
                continue
            if full or pno == 1 or (acc, pno) in cited:
                pages.append(pg.model_dump())
        if full:
            total = sum(len(p["segments"][0]["text"]) for p in pages)
            print(desc, "pages", len(pages), "approx chars", total)
        payload = {
            "meta": meta.model_dump(),
            "doc_urls": {k: v for k, v in corpus.doc_urls[acc].items()},
            "pages": pages,
        }
        with gzip.open(out / f"{acc}.json.gz", "wt") as fh:
            json.dump(payload, fh)
    # golden questions answerable from the sample, plus the unanswerable ones
    have = {
        (p["accession"], p["document"], p["page"])
        for f in out.glob("*.json.gz")
        for p in json.load(gzip.open(f, "rt"))["pages"]
    }
    sample_q = [
        r
        for r in load_golden()
        if (
            r["answerable"]
            and r["evidence"]
            and all((e["accession"], e["document"], e["page"]) in have for e in r["evidence"])
        )
        or (not r["answerable"] and r["id"] in {"u01", "u09", "u12", "u19", "u26"})
    ]
    (config.GOLDEN_DIR / "sample_golden.jsonl").write_text("\n".join(json.dumps(r) for r in sample_q) + "\n")
    print(len(sample_q), "sample questions", file=sys.stderr)


if __name__ == "__main__":
    main()
