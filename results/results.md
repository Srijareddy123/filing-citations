**Retrieval only (no model)**

| Configuration | Evidence recall@8 | All evidence found@8 | MRR | Stale hits (version traps) | Retrieval p50 / p95 |
|---|---|---|---|---|---|
| BM25 only | 0.66 | 0.65 | 0.53 | 17% | 10 / 20 ms |
| Dense only | 0.61 | 0.59 | 0.45 | 22% | 36 / 56 ms |
| Hybrid (RRF) | 0.72 | 0.68 | 0.51 | 20% | 44 / 53 ms |
| Hybrid + rerank | 0.65 | 0.59 | 0.53 | 15% | 297 / 375 ms |
| Hybrid + rerank + version filter | 0.65 | 0.59 | 0.53 | 10% | 307 / 370 ms |

**Answers**

Model: `ollama:qwen2.5:1.5b`

| Configuration | Answer correctness | Citation precision | Quote verbatim | Refusal accuracy (unanswerable) | False refusals | p50 / p95 |
|---|---|---|---|---|---|---|
| BM25 only | 0.33 | 0.51 | 1.00 | 0.44 | 0.30 | 6.0 / 8.9 s |
| Dense only | 0.32 | 0.54 | 1.00 | 0.62 | 0.39 | 6.7 / 10.8 s |
| Hybrid (RRF) | 0.38 | 0.53 | 1.00 | 0.66 | 0.27 | 6.7 / 10.5 s |
| Hybrid + rerank | 0.35 | 0.61 | 1.00 | 0.59 | 0.34 | 7.0 / 11.0 s |
| Hybrid + rerank + version filter | 0.34 | 0.63 | 1.00 | 0.62 | 0.32 | 6.9 / 11.4 s |
