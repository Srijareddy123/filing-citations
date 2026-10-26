"""FastAPI app: ask a question, open a cited page with the quote highlighted."""

from __future__ import annotations

import html
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config
from .agent import FilingAgent
from .ingest import Corpus
from .llm import get_llm
from .models import Answer, SearchFilters
from .retrieval import CONFIGS, default_retriever
from .service import Services

WEB = Path(__file__).resolve().parents[2] / "web"


class AskRequest(BaseModel):
    question: str
    config: str = "hybrid_rerank_version"
    ticker: str | None = None
    form: str | None = None
    filed_after: str | None = None
    filed_before: str | None = None


def highlight(text: str, quote: str) -> str:
    """HTML-escape text and wrap the quote in <mark>, matching across whitespace differences."""
    words = quote.split()
    if not words:
        return html.escape(text)
    pat = re.compile(r"\s+".join(re.escape(w) for w in words), re.I)
    m = pat.search(text)
    if not m:
        return html.escape(text)
    return (
        html.escape(text[: m.start()])
        + '<mark id="quote">'
        + html.escape(m.group(0))
        + "</mark>"
        + html.escape(text[m.end() :])
    )


def create_app(services: Services | None = None, llm=None) -> FastAPI:
    app = FastAPI(title="filing-citations")
    state: dict = {}

    def get_services() -> Services:
        if "svc" not in state:
            state["svc"] = services or Services(default_retriever(), Corpus.load(config.PARSED_DIR))
        return state["svc"]

    def get_model():
        if "llm" not in state:
            state["llm"] = llm or get_llm()
        return state["llm"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB / "index.html").read_text()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "configs": list(CONFIGS)}

    @app.post("/api/ask", response_model=Answer)
    def ask(req: AskRequest) -> Answer:
        if req.config not in CONFIGS:
            raise HTTPException(400, f"config must be one of {list(CONFIGS)}")
        s = get_services()
        agent = FilingAgent(s.retriever, s.corpus, get_model(), config_name=req.config)
        f = None
        if req.ticker or req.form or req.filed_after or req.filed_before:
            f = SearchFilters(
                tickers=[req.ticker.upper()] if req.ticker else [],
                forms=[req.form.upper()] if req.form else [],
                filed_after=req.filed_after,
                filed_before=req.filed_before,
                current_only=CONFIGS[req.config]["current"],
            )
        return agent.ask(req.question, f)

    @app.get("/view", response_class=HTMLResponse)
    def view(accession: str, page: int, document: str = "main", quote: str = "") -> str:
        s = get_services()
        pg = s.corpus.page(accession, document, page)
        if pg is None:
            raise HTTPException(404, "page not found")
        m = s.corpus.metas[accession]
        body = "".join(
            f"<h3>{html.escape(seg.section_title)}</h3><p>{highlight(seg.text, quote).replace(chr(10), '<br>')}</p>"
            for seg in pg.segments
        )
        src = html.escape(s.corpus.page_url(accession, document))
        head = f"{html.escape(m.company)} {html.escape(m.form)} filed {m.filed} ({m.version}), page {page}"
        tpl = (WEB / "page.html").read_text()
        return tpl.replace("{{HEAD}}", head).replace("{{SRC}}", src).replace("{{BODY}}", body)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run("filingcite.api:create_app", factory=True, host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
