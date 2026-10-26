"""Turn results/*.json into markdown tables: a compact one for the README, the full set for docs/RESULTS.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .evaluate import LABELS

START, END = "<!-- results:start -->", "<!-- results:end -->"


def retrieval_table(data: dict) -> str:
    head = (
        "| Configuration | Evidence recall@8 | All evidence found@8 | MRR | Stale hits (version traps) | "
        "Retrieval p50 / p95 |\n|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {LABELS[k]} | {v['evidence_recall_at_8']:.2f} | {v['all_evidence_found_at_8']:.2f} | {v['mrr']:.2f} | "
        f"{v['stale_hit_share_version_traps']:.0%} | {v['retrieval_p50_ms']} / {v['retrieval_p95_ms']} ms |"
        for k, v in data.items()
    ]
    return head + "\n".join(rows)


def answer_table(data: dict) -> str:
    model = next(iter(data.values()))["model"]
    head = (
        f"Model: `{model}`\n\n| Configuration | Answer correctness | Citation precision | Quote verbatim | "
        "Refusal accuracy (unanswerable) | False refusals | p50 / p95 |\n|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {LABELS[k]} | {v['answer_correctness']:.2f} | {v['citation_precision']:.2f} | "
        f"{v['quote_verbatim_rate']:.2f} | {v['refusal_accuracy_unanswerable']:.2f} | "
        f"{v['false_refusal_rate_answerable']:.2f} | {v['p50_s']:.1f} / {v['p95_s']:.1f} s |"
        for k, v in data.items()
    ]
    return head + "\n".join(rows)


def compact_table(data: dict) -> str:
    head = (
        "| Configuration | Answer correctness | Citation precision | Refusal accuracy | p50 / p95 |\n"
        "|---|---|---|---|---|\n"
    )
    rows = [
        f"| {LABELS[k]} | {v['answer_correctness']:.2f} | {v['citation_precision']:.2f} | "
        f"{v['refusal_accuracy_unanswerable']:.2f} | {v['p50_s']:.1f} / {v['p95_s']:.1f} s |"
        for k, v in data.items()
    ]
    return head + "\n".join(rows)


def build_compact(results: Path = config.RESULTS_DIR) -> str:
    files = sorted(results.glob("answers_*.json"))
    return "\n\n".join(compact_table(json.loads(f.read_text())) for f in files)


def replace_block(path: Path, text: str) -> None:
    if not path.exists() or START not in path.read_text():
        return
    new = re.sub(
        f"{re.escape(START)}.*?{re.escape(END)}", lambda _: f"{START}\n\n{text}\n\n{END}", path.read_text(), flags=re.S
    )
    path.write_text(new)


def build(results: Path = config.RESULTS_DIR) -> str:
    parts = []
    if (results / "retrieval.json").exists():
        parts.append(
            "**Retrieval only (no model)**\n\n" + retrieval_table(json.loads((results / "retrieval.json").read_text()))
        )
    for f in sorted(results.glob("answers_*.json")):
        parts.append("**Answers**\n\n" + answer_table(json.loads(f.read_text())))
    return "\n\n".join(parts)


def main() -> None:
    text = build()
    (config.RESULTS_DIR / "results.md").write_text(text + "\n")
    replace_block(config.ROOT / "README.md", build_compact())
    replace_block(config.ROOT / "docs" / "RESULTS.md", text)
    print(text)


if __name__ == "__main__":
    main()
