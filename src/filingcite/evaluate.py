"""Evaluate the five retrieval configurations on the golden set.

python -m filingcite.evaluate retrieval            retrieval-only metrics, no LLM
python -m filingcite.evaluate answers [--provider ollama] [--limit N]   answer metrics with a model
python -m filingcite.evaluate report               write results/results.md from saved runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from . import config
from .agent import FilingAgent
from .golden import load_golden
from .ingest import Corpus
from .llm import get_llm
from .retrieval import CONFIGS, default_retriever

RESULTS = config.RESULTS_DIR
TOP_K = 8
LABELS = {
    "bm25": "BM25 only",
    "dense": "Dense only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
    "hybrid_rerank_version": "Hybrid + rerank + version filter",
}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def norm_num(s: str) -> str:
    return re.sub(r"[\s,$]", "", s.lower())


def is_correct(answer: str, keys: list[list[str]]) -> bool:
    a = norm_num(answer)
    return bool(keys) and all(any(norm_num(alt) in a for alt in group) for group in keys)


def gold_pages(rec: dict) -> set[tuple[str, str, int]]:
    return {(e["accession"], e["document"], e["page"]) for e in rec["evidence"]}


def retrieval_run(retriever, golden: list[dict], cfg: str) -> list[dict]:
    retriever.search("warm up", cfg, TOP_K)
    rows = []
    for rec in golden:
        t0 = time.perf_counter()
        hits = retriever.search(rec["question"], cfg, TOP_K)
        dt = time.perf_counter() - t0
        row = {"id": rec["id"], "type": rec["type"], "latency_s": dt}
        got = [(h.chunk.accession, h.chunk.document, h.chunk.page) for h in hits]
        row["stale"] = sum(not h.chunk.current for h in hits)
        row["n_hits"] = len(hits)
        if rec["answerable"]:
            gold = gold_pages(rec)
            found = gold & set(got)
            row["recall"] = len(found) / len(gold)
            row["all_found"] = len(found) == len(gold)
            row["rr"] = next((1 / (i + 1) for i, g in enumerate(got) if g in gold), 0.0)
        rows.append(row)
    return rows


def summarize_retrieval(rows: list[dict]) -> dict:
    ans = [r for r in rows if "recall" in r]
    traps = [r for r in rows if r["type"] == "version_trap"]
    lat = [r["latency_s"] * 1000 for r in rows]
    return {
        "evidence_recall_at_8": round(statistics.mean(r["recall"] for r in ans), 3),
        "all_evidence_found_at_8": round(statistics.mean(r["all_found"] for r in ans), 3),
        "mrr": round(statistics.mean(r["rr"] for r in ans), 3),
        "stale_hit_share_version_traps": round(
            sum(r["stale"] for r in traps) / max(sum(r["n_hits"] for r in traps), 1), 3
        ),
        "retrieval_p50_ms": round(pct(lat, 0.5)),
        "retrieval_p95_ms": round(pct(lat, 0.95)),
        "n_answerable": len(ans),
    }


def cmd_retrieval(args) -> None:
    golden = load_golden()
    retriever = default_retriever()
    out = {}
    for cfg in CONFIGS:
        rows = retrieval_run(retriever, golden, cfg)
        out[cfg] = summarize_retrieval(rows)
        by_type: dict[str, list[float]] = {}
        for r in rows:
            if "recall" in r:
                by_type.setdefault(r["type"], []).append(r["recall"])
        out[cfg]["recall_by_type"] = {k: round(statistics.mean(v), 3) for k, v in by_type.items()}
        print(cfg, out[cfg], flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=2))


def cache_key(model: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()


class CachedLLM:
    """Disk cache so reruns and the same prompt across configs reuse model output."""

    def __init__(self, inner, path: Path) -> None:
        self.inner, self.path, self.name = inner, path, inner.name
        self.store: dict[str, str] = {}
        if path.exists():
            self.store = {k: v for k, v in (json.loads(x) for x in path.read_text().splitlines() if x)}

    def complete(self, system: str, user: str) -> str:
        key = cache_key(self.name, system, user)
        if key in self.store:
            return self.store[key]
        out = self.inner.complete(system, user)
        self.store[key] = out
        with self.path.open("a") as fh:
            fh.write(json.dumps([key, out]) + "\n")
        return out


def answer_run(agent: FilingAgent, golden: list[dict], resume: Path) -> list[dict]:
    done = {}
    if resume.exists():
        done = {r["id"]: r for r in (json.loads(x) for x in resume.read_text().splitlines() if x)}
    rows = []
    with resume.open("a") as fh:
        for rec in golden:
            if rec["id"] in done:
                rows.append(done[rec["id"]])
                continue
            a = agent.ask(rec["question"])
            gp = gold_pages(rec)
            cits = a.citations
            good = [c for c in cits if c.verified and (c.accession, c.document, c.page) in gp]
            row = {
                "id": rec["id"],
                "type": rec["type"],
                "answerable": rec["answerable"],
                "refused": not a.answerable,
                "correct": a.answerable and is_correct(a.answer, rec["keys"]) if rec["answerable"] else False,
                "n_cites": len(cits),
                "n_good_cites": len(good),
                "n_verified": sum(c.verified for c in cits),
                "latency_s": a.latency_s,
                "answer": a.answer[:300],
                "model": a.model,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
    return rows


def summarize_answers(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    cited = [r for r in ans if not r["refused"]]
    n_cites = sum(r["n_cites"] for r in cited)
    lat = [r["latency_s"] for r in rows]
    return {
        "answer_correctness": round(statistics.mean(r["correct"] for r in ans), 3),
        "citation_precision": round(sum(r["n_good_cites"] for r in cited) / n_cites, 3) if n_cites else 0.0,
        "quote_verbatim_rate": round(sum(r["n_verified"] for r in cited) / n_cites, 3) if n_cites else 0.0,
        "refusal_accuracy_unanswerable": round(statistics.mean(r["refused"] for r in un), 3),
        "false_refusal_rate_answerable": round(statistics.mean(r["refused"] for r in ans), 3),
        "p50_s": round(pct(lat, 0.5), 2),
        "p95_s": round(pct(lat, 0.95), 2),
        "n": len(rows),
        "model": rows[0]["model"] if rows else "",
    }


def cmd_answers(args) -> None:
    golden = load_golden()
    if args.limit:
        golden = golden[:: max(len(golden) // args.limit, 1)][: args.limit]
    corpus = Corpus.load(config.PARSED_DIR)
    retriever = default_retriever()
    RESULTS.mkdir(exist_ok=True)
    inner = get_llm(args.provider)
    llm = CachedLLM(inner, RESULTS / f"llm_cache_{re.sub(r'[^a-z0-9]+', '_', inner.name)}.jsonl")
    out = {}
    for cfg in args.configs or CONFIGS:
        agent = FilingAgent(retriever, corpus, llm, config_name=cfg)
        tag = re.sub(r"[^a-z0-9]+", "_", llm.name)
        rows = answer_run(agent, golden, RESULTS / f"answers_{cfg}_{tag}.jsonl")
        out[cfg] = summarize_answers(rows)
        print(cfg, out[cfg], flush=True)
    (RESULTS / f"answers_{re.sub(r'[^a-z0-9]+', '_', llm.name)}.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("retrieval")
    a = sub.add_parser("answers")
    a.add_argument("--provider", default=None)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    {"retrieval": cmd_retrieval, "answers": cmd_answers}[args.cmd](args)


if __name__ == "__main__":
    main()
