"""Switchable chat model: Ollama (local), Anthropic API, Amazon Bedrock, or an offline extractive stub."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Protocol

import httpx

from . import config


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        self.model = model or config.OLLAMA_MODEL
        self.url = url or config.OLLAMA_URL
        self.name = f"ollama:{self.model}"

    def complete(self, system: str, user: str) -> str:
        r = httpx.post(
            f"{self.url}/api/chat",
            timeout=300,
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 6144, "num_predict": 400},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class AnthropicLLM:
    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or config.ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


class BedrockLLM:
    def __init__(self, model: str | None = None) -> None:
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.model = model or config.BEDROCK_MODEL
        self.name = f"bedrock:{self.model}"

    def complete(self, system: str, user: str) -> str:
        r = self.client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0},
        )
        return r["output"]["message"]["content"][0]["text"]


_SRC = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.S)
_WORD = re.compile(r"[a-z0-9]{2,}")
_STOP = set(
    "the a an of in on for to and or is are was were what which who how much many does did do "
    "its it by with as at from that this be".split()
)


_MONTH = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"
)
_PROPER = re.compile(r"(?<!^)(?<![.?!] )\b([A-Z][A-Za-z&]{2,})")


class ExtractiveLLM:
    """No model: answers with the passage sentence that best overlaps the question, or refuses.

    It refuses when a proper noun in the question appears in none of the passages, which is how the demo
    handles companies that are not in the index.
    """

    name = "extractive"

    def complete(self, system: str, user: str) -> str:
        passages, question = user.split("QUESTION:")[0], user.split("QUESTION:")[-1].strip()
        q = question.lower()
        names = [w for w in _PROPER.findall(question) if w not in {"What", "How", "Which", "Who", "When", "The"}]
        if any(n.lower() not in passages.lower() for n in names):
            return json.dumps({"answerable": False, "answer": "", "citations": []})
        qw = {w for w in _WORD.findall(q) if w not in _STOP}
        want_date = bool(re.search(r"\bdate\b|\bwhen\b", q))
        want_num = bool(re.search(r"how many|how much|what (was|were|is) the (total|number|size|price)", q))
        sents = [
            (int(m.group(1)), sent.strip())
            for m in _SRC.finditer(passages)
            for sent in re.split(r"(?<=[.;])\s+|\n", m.group(2))
            if len(sent) >= 30
        ]
        sets = [set(_WORD.findall(sent.lower())) for _, sent in sents]
        idf = {w: math.log(1 + len(sets) / (1 + sum(w in sw for sw in sets))) for w in qw}
        total = sum(idf.values()) or 1.0
        best = (0.0, 0, "")
        for (src, sent), sw in zip(sents, sets, strict=True):
            score = sum(idf[w] for w in qw & sw) / total / (1 + len(sw) / 60)
            score += 0.4 * (want_date and bool(_MONTH.search(sent))) + 0.15 * (
                want_num and bool(re.search(r"\d", sent))
            )
            if score > best[0]:
                best = (score, src, sent)
        if best[0] < 0.3:
            return json.dumps({"answerable": False, "answer": "", "citations": []})
        return json.dumps({"answerable": True, "answer": best[2], "citations": [{"source": best[1], "quote": best[2]}]})


def get_llm(provider: str | None = None) -> LLM:
    p = provider or config.LLM_PROVIDER
    if p == "ollama":
        return OllamaLLM()
    if p == "anthropic":
        return AnthropicLLM()
    if p == "bedrock":
        return BedrockLLM()
    if p == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown provider {p}")
