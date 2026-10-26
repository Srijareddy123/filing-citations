"""Fetch filings from SEC EDGAR with fair-access limits (declared User-Agent, under 10 requests/s)."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from . import config

TICKERS = (
    "TSLA DIS AMD MRK DASH EBAY SHOP AAPL MSFT NVDA AMZN META KO PEP WMT HD NKE NFLX INTC ORCL BA CAT UBER SBUX PYPL F"
).split()
MAX_DOC_BYTES = 3_000_000
EIGHT_K_ITEMS = ("1.01", "2.02", "2.03", "5.02", "8.01")


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
            self.next_ok = max(now, self.next_ok) + self.interval


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or config.SEC_USER_AGENT
        if not ua:
            raise SystemExit("Set SEC_USER_AGENT in .env (see .env.example). SEC requires a name and email.")
        self.http = httpx.Client(headers={"User-Agent": ua}, timeout=60, follow_redirects=True)
        self.limiter = RateLimiter(config.SEC_MAX_RPS)

    def get(self, url: str) -> httpx.Response:
        for attempt in range(4):
            self.limiter.wait()
            r = self.http.get(url)
            if r.status_code in (429, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"gave up on {url}")

    def submissions(self, cik: int) -> dict:
        return self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()

    def index_rows(self, cik: int, accession: str) -> list[dict]:
        """Documents in a filing: filename, type, description, size."""
        nodash = accession.replace("-", "")
        r = self.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accession}-index.htm")
        soup = BeautifulSoup(r.text, "lxml")
        rows = []
        for tr in soup.select("table.tableFile tr"):
            tds = tr.find_all("td")
            if len(tds) < 5 or not tds[2].find("a"):
                continue
            href = tds[2].find("a")["href"]
            name = href.rsplit("/", 1)[-1]
            rows.append(
                {
                    "filename": name,
                    "type": tds[3].get_text(strip=True),
                    "desc": tds[1].get_text(" ", strip=True),
                    "size": tds[4].get_text(strip=True),
                }
            )
        return rows


def _recent(sub: dict) -> list[dict]:
    rec = sub["filings"]["recent"]
    keys = rec.keys()
    return [{k: rec[k][i] for k in keys} for i in range(len(rec["form"]))]


def select_filings(sub: dict, client: EdgarClient | None = None) -> list[dict]:
    """Pick the filings to ingest for one company."""
    rows = _recent(sub)
    tenks = sorted((r for r in rows if r["form"] == "10-K"), key=lambda r: r["reportDate"], reverse=True)
    amends = [r for r in rows if r["form"] == "10-K/A" and r["filingDate"] >= "2023-06-01"]
    keep_periods = {r["reportDate"] for r in tenks[:2]} | {r["reportDate"] for r in amends}
    chosen = [r for r in tenks if r["reportDate"] in keep_periods] + amends
    last_k = tenks[0]["filingDate"] if tenks else "2024-01-01"
    qs = sorted(
        (r for r in rows if r["form"] == "10-Q" and r["filingDate"] > last_k),
        key=lambda r: r["filingDate"],
        reverse=True,
    )
    if len(qs) < 2:  # fall back to the most recent two 10-Qs of any date
        qs = sorted((r for r in rows if r["form"] == "10-Q"), key=lambda r: r["filingDate"], reverse=True)
    chosen += qs[:2]
    eights = sorted(
        (
            r
            for r in rows
            if r["form"] == "8-K"
            and r["filingDate"] >= "2024-06-01"
            and any(i in r.get("items", "") for i in EIGHT_K_ITEMS)
        ),
        key=lambda r: r["filingDate"],
        reverse=True,
    )
    chosen += _pick_eight_ks(eights, int(sub["cik"]), client)
    return chosen


def _pick_eight_ks(candidates: list[dict], cik: int, client: EdgarClient | None) -> list[dict]:
    """Up to 3 8-Ks, preferring one with a material-agreement exhibit and one earnings release."""
    if client is None:
        return candidates[:3]
    picked: list[dict] = []
    want = {"agreement": False, "release": False}
    for r in candidates[:14]:
        if len(picked) >= 3:
            break
        types = [d["type"] for d in client.index_rows(cik, r["accessionNumber"])]
        has10 = any(t.startswith("EX-10") for t in types)
        has99 = any(t.startswith("EX-99") for t in types)
        if has10 and not want["agreement"]:
            want["agreement"] = True
            picked.append(r)
        elif has99 and not want["release"] and "2.02" in r["items"]:
            want["release"] = True
            picked.append(r)
        elif (has10 or has99) and len(picked) < 3 and want["agreement"] and want["release"]:
            picked.append(r)
    return picked


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def download_filing(client: EdgarClient, ticker: str, sub: dict, r: dict) -> dict | None:
    cik, acc = int(sub["cik"]), r["accessionNumber"]
    nodash = acc.replace("-", "")
    out_dir = config.RAW_DIR / ticker / acc
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    index = client.index_rows(cik, acc)
    wanted = []
    for d in index:
        t = d["type"]
        if t == r["form"] or (r["form"] == "8-K" and t in ("8-K",)):
            wanted.append((d, "main"))
        elif r["form"] == "8-K" and (t.startswith("EX-10") or t.startswith("EX-99")):
            wanted.append((d, t))
    wanted = wanted[:4]
    for d, doc_type in wanted:
        if not d["filename"].lower().endswith((".htm", ".html")):
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{d['filename']}"
        path = out_dir / _safe(d["filename"])
        if not path.exists():
            resp = client.get(url)
            if len(resp.content) > MAX_DOC_BYTES * 8:
                continue
            path.write_bytes(resp.content)
        docs.append({"document": doc_type, "filename": path.name, "desc": d["desc"], "url": url})
    if not docs:
        return None
    return {
        "ticker": ticker,
        "cik": cik,
        "company": sub["name"],
        "accession": acc,
        "form": r["form"],
        "filed": r["filingDate"],
        "period": r["reportDate"],
        "items": r.get("items", ""),
        "docs": docs,
    }


def fetch_all(tickers: list[str] | None = None) -> Path:
    client = EdgarClient()
    tmap = {v["ticker"]: v for v in client.get("https://www.sec.gov/files/company_tickers.json").json().values()}
    manifest = config.RAW_DIR / "manifest.jsonl"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest.exists():
        done = {json.loads(line)["accession"] for line in manifest.read_text().splitlines() if line}
    with manifest.open("a") as fh:
        for tk in tickers or TICKERS:
            cik = int(tmap[tk]["cik_str"])
            sub = client.submissions(cik)
            for r in select_filings(sub, client):
                if r["accessionNumber"] in done:
                    continue
                rec = download_filing(client, tk, sub, r)
                if rec:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(
                        f"{tk} {rec['form']:6} {rec['filed']} {rec['accession']} docs={len(rec['docs'])}",
                        flush=True,
                    )
    return manifest


if __name__ == "__main__":
    fetch_all()
