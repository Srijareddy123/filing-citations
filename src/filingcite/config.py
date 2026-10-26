"""Paths and settings. Secrets and local overrides come from the environment or .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("FC_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SAMPLE_DIR = ROOT / "data" / "sample"
GOLDEN_DIR = ROOT / "data" / "golden"
RESULTS_DIR = ROOT / "results"

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9201")
INDEX_NAME = os.getenv("FC_INDEX", "filings")

# Retrieval backend: "opensearch" (full path) or "memory" (tests, no Docker).
BACKEND = os.getenv("FC_BACKEND", "opensearch")
EMBED_MODEL = os.getenv("FC_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("FC_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM provider: ollama | anthropic | bedrock | extractive (offline, no model)
LLM_PROVIDER = os.getenv("FC_LLM", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
SEC_MAX_RPS = float(os.getenv("SEC_MAX_RPS", "8"))  # SEC allows 10, stay under

CHUNK_CHARS = 1400
