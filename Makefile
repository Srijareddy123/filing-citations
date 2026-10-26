PY := .venv/bin/python
export OPENSEARCH_URL ?= http://localhost:9201

.PHONY: setup fetch parse up index demo demo-lite ui test lint eval eval-retrieval report down clean-data sample golden

setup:
	python3.13 -m venv .venv || python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e . --no-deps

# Full corpus: needs SEC_USER_AGENT in .env
fetch:
	$(PY) -m filingcite.edgar
parse:
	$(PY) -m filingcite.ingest
up:
	docker compose up -d --wait
index: up
	$(PY) -m filingcite.index

# Demo on the committed sample filings: OpenSearch in Docker, a local Ollama model if running, else an offline stub.
demo: up
	FC_INDEX=filings_demo FC_DATA_DIR=data $(PY) -m filingcite.index data/sample
	FC_INDEX=filings_demo $(PY) -m filingcite.cli demo
demo-lite:
	FC_BACKEND=memory FC_EMBED_MODEL=hash FC_RERANK_MODEL=lexical $(PY) -m filingcite.cli demo --provider extractive
ui:
	$(PY) -m filingcite.api

test:
	$(PY) -m pytest -q
lint:
	.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts

# Retrieval metrics (no model needed), then answer metrics with the local model
eval-retrieval:
	$(PY) -m filingcite.evaluate retrieval
eval: eval-retrieval
	$(PY) -m filingcite.evaluate answers --provider ollama
report:
	$(PY) -m filingcite.report

sample:
	$(PY) scripts/make_sample.py
golden:
	$(PY) -m filingcite.golden build
down:
	docker compose down -v
