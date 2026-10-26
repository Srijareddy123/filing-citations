"""Parse EDGAR HTML into pages and sections.

Pages follow the page breaks in the filing HTML. When a document has almost none
(some amendments), it is cut into synthetic pages of about 3500 characters.
Sections come from "Item N." headings in 10-K and 10-Q, "Item N.NN" in 8-K, and
"ARTICLE" headings in exhibits.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, XMLParsedAsHTMLWarning

from .models import FilingMeta, Page, Segment

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BREAK = "\f"
BLOCKS = {
    "p",
    "div",
    "tr",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "section",
    "center",
    "ul",
    "ol",
    "body",
    "html",
    "tbody",
    "thead",
    "blockquote",
    "article",
}
SKIP = {"script", "style", "head", "title", "ix:header"}
BREAK_RE = re.compile(r"page-break-(after|before)\s*:\s*always|break-(after|before)\s*:\s*page", re.I)
BREAK_BEFORE_RE = re.compile(r"(page-break-before\s*:\s*always|break-before\s*:\s*page)", re.I)
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[A-C]?)\s*[\.\-:–—]?\s*(.*)$", re.I)
ITEM8K_RE = re.compile(r"^item\s+(\d\.\d{2})\s*[\.\-:–—]?\s*(.*)$", re.I)
PART_RE = re.compile(r"^part\s+(i{1,3}v?|iv)\b[\s\.\-:–—]*(.*)$", re.I)
ARTICLE_RE = re.compile(r"^article\s+([ivxlc\d]+)\b[\s\.\-:–—]*(.*)$", re.I)
INLINE_RE = re.compile(r"display\s*:\s*inline(?!-)", re.I)
SYNTH_PAGE_CHARS = 3500
MIN_CHARS_PER_BREAK = 25000


def _clean(s: str) -> str:
    s = re.sub("[\u2002\u2003\u2009\u202f\u2007]", " ", s).replace("\xa0", " ").replace("​", "")
    return re.sub(r"[ \t\r\n]+", " ", s).strip()


def _merge_cells(cells: list[str]) -> str:
    out: list[str] = []
    for c in cells:
        if not c:
            continue
        if out and (c in (")", "%", ")%") or c.startswith(")")):
            out[-1] += c
        elif out and out[-1] in ("$", "(", "$("):
            out[-1] += c
        else:
            out.append(c)
    return " | ".join(out)


def _hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _cell_text(cell: Tag) -> str:
    return _clean(" ".join(_walk(cell)))


def _walk(node: Tag) -> Iterator[str]:
    """Yield text lines in document order, with BREAK markers for page breaks.

    Consecutive inline content (text, spans, links) is joined without extra spaces so
    words split across spans stay whole; block elements start a new line.
    """
    buf: list[str] = []

    def flush() -> Iterator[str]:
        line = _clean("".join(buf))
        buf.clear()
        if line:
            yield line

    for child in node.children:
        if isinstance(child, NavigableString):
            if not isinstance(child, Comment):
                buf.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in SKIP or _hidden(child):
            continue
        style = child.get("style") or ""
        is_break = bool(BREAK_RE.search(style)) or name == "hr" and "page" in style.lower()
        if is_break and BREAK_BEFORE_RE.search(style):
            yield from flush()
            yield BREAK
        if name == "br":
            buf.append(" ")
        elif name == "tr":
            yield from flush()
            line = _merge_cells([_cell_text(td) for td in child.find_all(["td", "th"])])
            if line:
                yield line
        elif INLINE_RE.search(style) and not child.find(list(BLOCKS)):
            buf.append(child.get_text(""))
        elif name in BLOCKS or child.find(list(BLOCKS)):
            yield from flush()
            yield from _walk(child)
        else:
            buf.append(child.get_text(""))
        if is_break and not BREAK_BEFORE_RE.search(style):
            yield from flush()
            yield BREAK
    yield from flush()


def html_to_lines(html: bytes | str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup
    lines = [ln for ln in _walk(root) if ln]
    return lines


def split_pages(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = [[]]
    for ln in lines:
        if ln == BREAK:
            pages.append([])
        else:
            pages[-1].append(ln)
    return pages


def synthetic_pages(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = [[]]
    size = 0
    for ln in lines:
        if ln == BREAK:
            continue
        if size + len(ln) > SYNTH_PAGE_CHARS and pages[-1]:
            pages.append([])
            size = 0
        pages[-1].append(ln)
        size += len(ln)
    return pages


class SectionTracker:
    """Assign each line to a section based on headings seen so far."""

    def __init__(self, form: str, document: str) -> None:
        self.form = form
        self.document = document
        self.part = ""
        self.key = "cover" if document == "main" else "exhibit"
        self.title = "Cover page" if document == "main" else document

    def feed(self, line: str) -> None:
        m2 = re.match(r"^((?:item|part)\s+[0-9A-Za-z.]+)\s*\|\s*([^|]+)$", line, re.I)
        if m2 and not m2.group(2).strip().isdigit():
            line = f"{m2.group(1)} {m2.group(2)}"
        if len(line) > 170 or " | " in line:
            return
        base = self.form.split("/")[0]
        m = PART_RE.match(line)
        if m and base in ("10-K", "10-Q") and len(line) < 60:
            self.part = m.group(1).lower()
            if base == "10-K" and not m.group(2):
                return
        if base in ("10-K", "10-Q") and self.document == "main":
            m = ITEM_RE.match(line)
            if m and (m.group(2) or len(line) < 12):
                num = m.group(1).lower()
                key = f"item_{num}"
                if base == "10-Q" and self.part:
                    key = f"part_{self.part}_{key}"
                self.key = key
                self.title = f"Item {m.group(1).upper()}. {m.group(2).strip()}".strip(" .")
        elif base == "8-K" and self.document == "main":
            m = ITEM8K_RE.match(line)
            if m:
                self.key = "item_" + m.group(1).replace(".", "_")
                self.title = f"Item {m.group(1)}. {m.group(2).strip()}".strip(" .")
        elif self.document != "main":
            m = ARTICLE_RE.match(line)
            if m:
                self.key = f"article_{m.group(1).lower()}"
                self.title = f"{self.document} Article {m.group(1).upper()} {m.group(2).strip()}".strip()


def parse_document(html: bytes | str, meta: FilingMeta, document: str) -> list[Page]:
    lines = html_to_lines(html)
    total = sum(len(x) for x in lines if x != BREAK)
    pages_lines = split_pages(lines)
    n_breaks = len(pages_lines) - 1
    source = "break"
    if n_breaks == 0 or total / max(n_breaks, 1) > MIN_CHARS_PER_BREAK:
        pages_lines = synthetic_pages(lines)
        source = "synthetic"
    tracker = SectionTracker(meta.form, document)
    pages: list[Page] = []
    for i, pl in enumerate(pages_lines, start=1):
        segs: list[Segment] = []
        for ln in pl:
            tracker.feed(ln)
            if segs and segs[-1].section == tracker.key and segs[-1].section_title == tracker.title:
                segs[-1].text += "\n" + ln
            else:
                segs.append(Segment(section=tracker.key, section_title=tracker.title, text=ln))
        if segs and sum(len(s.text) for s in segs) >= 20:
            pages.append(Page(accession=meta.accession, document=document, page=i, segments=segs, page_source=source))  # type: ignore[arg-type]
    return pages
