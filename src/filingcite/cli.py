"""Command line: python -m filingcite.cli ask "question" | demo"""

from __future__ import annotations

import argparse
import sys

import httpx

from . import config
from .agent import FilingAgent
from .ingest import Corpus
from .llm import get_llm
from .retrieval import CONFIGS, Retriever, default_retriever

DEMO_QUESTIONS = [
    "What date does Merck's current 10-K for fiscal 2023 give for its Annual Meeting of Shareholders?",
    "In AMD's original fiscal 2025 10-K, as first filed, what unit shipment increase was given for Client processors?",
    "What cash price per Delivery Hero share is Uber's takeover offer in the Business Combination Agreement?",
    "What was JPMorgan Chase's net interest income in 2025?",
]


def pick_llm(provider: str | None):
    """Use the requested provider. For the default (ollama), fall back to the offline stub when Ollama is down."""
    p = provider or config.LLM_PROVIDER
    if p == "ollama":
        try:
            httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).raise_for_status()
        except httpx.HTTPError:
            print("Ollama is not running, using the offline extractive stub instead.", file=sys.stderr)
            return get_llm("extractive")
    return get_llm(p)


def show(ans) -> None:
    print(f"\nQ: {ans.question}")
    print(f"A: {ans.answer}" + ("" if ans.answerable else f"   (refused: {ans.refusal_reason})"))
    for i, c in enumerate(ans.citations, 1):
        print(
            f"  [{i}] {c.company} {c.form} filed {c.filed} ({c.version}), {c.section_title or c.section}, page {c.page}"
        )
        print(f'      "{c.quote[:200]}"')
    print(f"  model={ans.model} latency={ans.latency_s:.1f}s")


def build_retriever(retriever_dir: str | None) -> tuple[Retriever, Corpus]:
    d = retriever_dir or str(config.PARSED_DIR)
    corpus = Corpus.load(__import__("pathlib").Path(d))
    return default_retriever(__import__("pathlib").Path(d)), corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--config", default="hybrid_rerank_version", choices=list(CONFIGS))
    a.add_argument("--provider")
    a.add_argument("--data")
    d = sub.add_parser("demo")
    d.add_argument("--provider")
    args = ap.parse_args()
    if args.cmd == "ask":
        r, corpus = build_retriever(args.data)
        show(FilingAgent(r, corpus, pick_llm(args.provider), args.config).ask(args.question))
    else:
        r, corpus = build_retriever(str(config.SAMPLE_DIR))
        agent = FilingAgent(r, corpus, pick_llm(args.provider))
        for q in DEMO_QUESTIONS:
            show(agent.ask(q))


if __name__ == "__main__":
    main()
