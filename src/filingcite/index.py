"""Embed chunks and load them into OpenSearch: python -m filingcite.index [parsed_dir]"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import config
from .backends import OpenSearchBackend
from .embed import get_embedder
from .ingest import Corpus, embed_text, make_chunks


def build(directory: Path) -> int:
    corpus = Corpus.load(directory)
    chunks = make_chunks(corpus)
    emb = get_embedder()
    be = OpenSearchBackend()
    be.create_index(emb.dim)
    t0 = time.time()
    step = 2000
    for i in range(0, len(chunks), step):
        part = chunks[i : i + step]
        be.add(part, emb.encode([embed_text(c) for c in part]))
        print(f"indexed {i + len(part)}/{len(chunks)} ({time.time() - t0:.0f}s)", flush=True)
    return len(chunks)


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else config.PARSED_DIR
    print(build(d), "chunks indexed from", d.name)
