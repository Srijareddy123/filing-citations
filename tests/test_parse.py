from filingcite.models import FilingMeta
from filingcite.parse import html_to_lines, parse_document

HTML = """<html><body>
<div>ANNUAL REPORT</div>
<div style="display:none">hidden header</div>
<div>Shares outstanding: <span>1,0</span><span>00</span> as of <div style="display:inline">May 1</div>, 2025.</div>
<div style="page-break-after:always"></div>
<p>PART I</p>
<p>Item 1A. Risk Factors</p>
<p>Competition may hurt results.</p>
<table><tr><td>Revenue</td><td>$</td><td>1,234</td><td>(5</td><td>)%</td></tr></table>
<hr style="page-break-after:always"/>
<p>Item 7. Management's Discussion</p>
<p>Sales grew.</p>
</body></html>"""


def meta(form="10-K"):
    return FilingMeta(
        accession="a", cik=1, ticker="X", company="X Inc", form=form, period="2025-12-31", filed="2026-02-01"
    )


def test_inline_spans_stay_whole_and_hidden_text_dropped():
    lines = html_to_lines(HTML)
    assert "Shares outstanding: 1,000 as of May 1, 2025." in lines
    assert not any("hidden" in x for x in lines)


def test_table_row_merges_currency_cells():
    assert "Revenue | $1,234 | (5)%" in html_to_lines(HTML)


def test_pages_and_sections():
    pages = parse_document(HTML, meta(), "main")
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[0].segments[0].section == "cover"
    assert pages[1].segments[-1].section == "item_1a"
    assert pages[2].segments[-1].section == "item_7"
    assert pages[1].page_source == "break"


def test_ten_q_sections_carry_the_part():
    html = "<p>PART II</p><p>Item 1A. Risk Factors</p><p>Nothing new.</p>"
    seg = parse_document(html, meta("10-Q"), "main")[0].segments[-1]
    assert seg.section == "part_ii_item_1a"


def test_eight_k_item_sections():
    html = "<p>Item 1.01 Entry into a Material Definitive Agreement.</p><p>Details of the deal here.</p>"
    seg = parse_document(html, meta("8-K"), "main")[0].segments[-1]
    assert seg.section == "item_1_01"


def test_synthetic_pages_when_no_breaks():
    html = "".join(f"<p>{'word ' * 120}paragraph {i}.</p>" for i in range(60))
    pages = parse_document(html, meta(), "main")
    assert len(pages) > 1 and pages[0].page_source == "synthetic"
