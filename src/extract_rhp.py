"""RHP PDF -> data/raw/<symbol>.json.

1. pdfplumber reads every page's text (cached in rhp/<symbol>.pages.json).
2. A keyword scorer picks the ~40 pages that matter (offer, capital structure, objects,
   restated financials, basis for offer price, ...), skipping table-of-contents pages.
3. Claude reads only those pages and returns figures *as printed* with their unit and
   page number. Unit conversion and every calculation happen in code, not in the model.
4. Sanity checks run on the result; failures are recorded, not hidden.

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .schema import SECTORS

MODEL = "claude-opus-5-5"
ROOT = Path(__file__).resolve().parents[1]
# RHP PDFs are large (~5 MB each). Set IPO_RHP_DIR to keep them outside a synced folder.
RHP_DIR = Path(os.environ.get("IPO_RHP_DIR") or ROOT / "rhp")
RAW_DIR = ROOT / "data" / "raw"
MAX_PAGES = 26

# --------------------------------------------------------------------------- #
# Extraction schema (what Claude returns)
# --------------------------------------------------------------------------- #
Unit = Literal["rupees", "thousand", "lakh", "million", "crore", "billion", "shares", "percent", "number"]
Sector = Literal[tuple(SECTORS)]  # type: ignore[valid-type]


class Fig(BaseModel):
    """One figure exactly as printed. value=null when absent or shown as [●]."""
    value: Optional[float]
    unit: Optional[Unit] = Field(description="Unit as printed. 'shares' for an absolute share count, 'rupees' for per-share prices/EPS/NAV, 'percent' for %.")
    page: Optional[int] = Field(description="Number from the '=== PAGE n ===' marker where the figure appears.")


class Seller(BaseModel):
    name: str
    kind: Literal["promoter", "promoter_group", "investor", "other"]
    shares_offered: Fig


class Proceed(BaseModel):
    purpose: str
    category: Literal["debt_repayment", "capex", "gcp", "other"]
    amount: Fig


class Peer(BaseModel):
    name: str
    nse_symbol: Optional[str]
    pe: Optional[float] = Field(description="P/E as printed in the peer table; null if 'NA' or negative/not meaningful.")
    ronw_pct: Optional[float]
    page: Optional[int]


class RHPExtract(BaseModel):
    company: Optional[str]
    sector: Optional[Sector]
    exchanges: Optional[Literal["NSE", "BSE", "both"]]
    price_low: Fig
    price_high: Fig
    lot_size: Fig
    fresh_issue_amount: Fig
    fresh_issue_shares: Fig
    ofs_amount: Fig
    ofs_shares: Fig
    selling_shareholders: list[Seller]
    pre_issue_shares: Fig = Field(description="Equity shares outstanding before the offer.")
    post_issue_shares: Fig = Field(description="Equity shares outstanding after the offer, if printed.")
    promoter_pre_pct: Fig = Field(description="Promoter + promoter group holding before the offer, % of pre-offer capital.")
    promoter_post_pct: Fig
    objects_of_offer: list[Proceed]
    retail_quota_pct: Fig
    pre_ipo_placement: Optional[bool] = Field(description="Any pre-IPO placement, preferential or rights allotment in the 12 months before the RHP.")
    pre_ipo_price: Fig = Field(description="Price per share of the most recent such allotment or secondary transfer.")
    lead_managers: list[str]
    financials_basis: Optional[Literal["consolidated", "standalone"]]
    fy0_label: Optional[str] = Field(description="Label of the latest full financial year, e.g. 'FY2024' or 'March 31, 2024'.")
    revenue_fy2: Fig
    revenue_fy1: Fig
    revenue_fy0: Fig
    ebitda_fy2: Fig
    ebitda_fy1: Fig
    ebitda_fy0: Fig
    pat_fy2: Fig
    pat_fy1: Fig
    pat_fy0: Fig
    cfo_fy2: Fig
    cfo_fy1: Fig
    cfo_fy0: Fig
    net_worth_fy0: Fig
    total_debt_fy0: Fig
    cash_fy0: Fig
    eps_fy0: Fig = Field(description="Basic EPS for fy0 in rupees.")
    ronw_fy0_pct: Fig
    nav_per_share_fy0: Fig
    stub_period_months: Optional[int]
    revenue_stub: Fig
    pat_stub: Fig
    no_listed_peers: bool
    peers: list[Peer]
    notes: list[str] = Field(description="Anything ambiguous: restatements, unit doubts, missing tables.")


SYSTEM = f"""You extract data from the Red Herring Prospectus (RHP) of an Indian IPO.
You receive selected pages; each starts with a marker '=== PAGE n ==='.

Rules:
- Copy every figure exactly as printed with the unit printed in that table's header
  (e.g. '₹ in million' -> unit 'million', '₹ in lakhs' -> 'lakh'). Do not convert units and do not
  calculate anything. Negative numbers (in brackets) are negative.
- page = the n of the marker of the page you read the figure from.
- value = null when a figure is not on these pages or is printed as [●]. Never guess and never write 0 for missing.
- Financials: use the RESTATED CONSOLIDATED statements (standalone only if there is no consolidated)
  and say which in financials_basis. fy0 = latest FULL financial year, fy1 the year before, fy2 two
  years before. A shorter interim period (e.g. 6 months to Sep 30) is the stub, not fy0.
- revenue = Revenue from operations (not total income). Look for it in the restated statement of
  profit and loss too, not only the summary. Only if no page given shows revenue from operations,
  use Total income instead and say so in notes. pat = restated profit for the year
  (after tax, including minority). cfo = net cash from operating activities.
  ebitda = EBITDA as reported in the RHP's key performance indicators; null if not reported.
  total_debt = total borrowings, current + non-current. net_worth = total equity attributable to owners.
- Shares: an absolute count uses unit 'shares'; if a table prints shares in lakh/million use that unit.
- selling_shareholders: every seller in the offer for sale with the shares each offers;
  kind = promoter / promoter_group / investor (PE, VC, corporate investors) / other (individuals etc.).
- objects_of_offer: each use of the FRESH issue proceeds (not the OFS), categorised as
  debt_repayment, capex (plant, machinery, expansion, new facilities), gcp (general corporate purposes), other.
- peers: the 'Comparison with listed industry peers' table in 'Basis for Offer Price'. Use the P/E and
  RoNW printed there, not current values. If it says there are no listed peers, set no_listed_peers=true.
- sector: choose the closest of the fixed list.
- Put any doubt (unit unclear, figures that don't reconcile, missing table) in notes."""


# --------------------------------------------------------------------------- #
# Page text and page selection
# --------------------------------------------------------------------------- #
SECTIONS: dict[str, tuple[int, list[str]]] = {   # name -> (pages to keep, keywords)
    "summary":    (4, ["summary of the offer document", "summary of financial information", "summary of restated", "summary of the offer"]),
    "offer":      (3, ["the offer", "fresh issue", "offer for sale", "selling shareholders", "employee reservation", "retail individual"]),
    "capital":    (5, ["capital structure", "shareholding pattern", "pre-offer", "post-offer", "promoter group", "pre-ipo", "weighted average cost of acquisition"]),
    "objects":    (3, ["objects of the offer", "net proceeds", "general corporate purposes", "repayment", "prepayment", "capital expenditure"]),
    "basis":      (3, ["basis for offer price", "listed industry peers", "listed peers", "p/e", "return on net worth", "price/earning"]),
    "financials": (9, ["restated consolidated statement of profit and loss", "restated consolidated statement of assets and liabilities",
                       "restated consolidated statement of cash flows", "revenue from operations", "net cash", "operating activities",
                       "ebitda", "key performance indicators", "total borrowings"]),
    "industry":   (2, ["industry overview", "our business", "we are a", "we are one of"]),
    "lead":       (1, ["book running lead managers", "brlm"]),
}


# Section headings as printed (upper case) near the top of the page where a section starts,
# and how many pages to take from there. Matched case-sensitively in the first ~200 characters.
HEADINGS: list[tuple[str, str, int]] = [
    # lean spans (trimmed 2026-11-19 to save plan tokens): ~22 pages/RHP instead of ~32
    ("summary_offer", r"SUMMARY OF (THE |THIS )?(OFFER DOCUMENT|RED HERRING PROSPECTUS|PROSPECTUS)|OFFER DOCUMENT SUMMARY", 4),
    ("the_offer",     r"(?<!OF )\bTHE (OFFER|ISSUE)\b(?! DOCUMENT| PRICE| RELATED)", 2),
    ("summary_fin",   r"SUMMARY (OF )?(RESTATED )?(CONSOLIDATED )?FINANCIAL (INFORMATION|STATEMENTS)", 4),
    ("capital",       r"CAPITAL STRUCTURE", 3),
    ("objects",       r"OBJECTS? OF THE (OFFER|ISSUE)", 2),
    ("basis",         r"BASIS FOR (THE )?(OFFER|ISSUE) PRICE", 3),
    ("other_fin",     r"OTHER FINANCIAL INFORMATION", 2),
    ("capitalisation", r"CAPITALI[SZ]ATION STATEMENT", 1),
]
# Full restated statements: used when there is no summary financial information section.
STATEMENTS = [r"restated (consolidated )?(summary )?statements? of (assets and liabilities|profits? and loss(es)?|cash flows?)"]
HEAD_CHARS = 220


def page_texts(pdf: Path) -> list[str]:
    """Text of every page (cached next to the PDF). PyMuPDF is ~50x faster than pdfplumber."""
    cache = pdf.with_suffix(".pages.json")
    if cache.exists() and cache.stat().st_mtime >= pdf.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        import pymupdf
        with pymupdf.open(pdf) as doc:
            pages = [pg.get_text() for pg in doc]
    except ImportError:
        import pdfplumber
        with pdfplumber.open(pdf) as doc:
            pages = [(p.extract_text() or "") for p in doc.pages]
    cache.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def is_toc(text: str) -> bool:
    return len(re.findall(r"\.{6,}", text)) >= 8 or "table of contents" in text.lower()[:400]


def _head(text: str) -> str:
    return " ".join(text.split())[:HEAD_CHARS]


def find_heading(pages: list[str], pattern: str, flags: int = 0, start: int = 3) -> int | None:
    """0-based index of the first page (after the cover/TOC) whose top carries the heading."""
    rx = re.compile(pattern, flags)
    for i in range(start, len(pages)):
        if pages[i].strip() and not is_toc(pages[i]) and rx.search(_head(pages[i])):
            return i
    return None


def _keyword_pages(pages: list[str], k: int, words: list[str]) -> list[int]:
    low = [p.lower() for p in pages]
    scored = [(sum(t.count(w) for w in words), -i, i) for i, t in enumerate(low) if t.strip() and not is_toc(t)]
    return [i for s, _, i in sorted(scored, reverse=True)[:k] if s]


def select_pages(pages: list[str], max_pages: int = MAX_PAGES) -> list[int]:
    """0-based indices of the pages to send, in document order.

    Each section is located by its heading; when a heading isn't found the section falls
    back to keyword scoring (SECTIONS)."""
    n = len(pages)
    chosen: list[int] = [i for i in range(min(2, n)) if pages[i].strip()]          # cover: issue summary, BRLMs
    found: dict[str, int] = {}
    for name, pattern, span in HEADINGS:
        i = find_heading(pages, pattern)
        if i is not None:
            found[name] = i
            chosen += list(range(i, min(i + span, n)))
    if "summary_fin" not in found:
        for pat in STATEMENTS:
            hits = [i for i in range(3, n) if pages[i].strip() and not is_toc(pages[i]) and re.search(pat, _head(pages[i]), re.I)]
            for i in hits[:6]:                         # first page of each statement + its continuation
                chosen += [i, i + 1] if i + 1 < n else [i]
            if hits:
                found["statements"] = hits[0]
    fallback = {"summary_offer": "summary", "the_offer": "offer", "capital": "capital", "objects": "objects",
                "basis": "basis", "summary_fin": "financials"}
    for name, sec in fallback.items():
        if name not in found and not (name == "summary_fin" and "statements" in found):
            k, words = SECTIONS[sec]
            chosen += _keyword_pages(pages, k, words)
            if sec == "financials":
                chosen += [i + 1 for i in _keyword_pages(pages, k, words) if i + 1 < n]
    seen: list[int] = []
    for i in chosen:
        if i not in seen:
            seen.append(i)
    return sorted(seen[:max_pages])


def build_prompt(pages: list[str], idx: list[int]) -> str:
    return "\n\n".join(f"=== PAGE {i + 1} ===\n{pages[i]}" for i in idx)


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5)   # rough; RHP English text with numbers


# --------------------------------------------------------------------------- #
# Units and flattening
# --------------------------------------------------------------------------- #
MULT = {"rupees": 1, "thousand": 1e3, "lakh": 1e5, "million": 1e6, "crore": 1e7, "billion": 1e9,
        "shares": 1, "number": 1, "percent": 1}


def absolute(f: Fig | None) -> float | None:
    """Rupees or share count as an absolute number."""
    if f is None or f.value is None:
        return None
    return f.value * MULT.get(f.unit or "number", 1)


def crore(f: Fig | None) -> float | None:
    a = absolute(f)
    return None if a is None else a / 1e7


def plain(f: Fig | None) -> float | None:
    return None if f is None else f.value


MONEY = ["revenue_fy2", "revenue_fy1", "revenue_fy0", "ebitda_fy2", "ebitda_fy1", "ebitda_fy0",
         "pat_fy2", "pat_fy1", "pat_fy0", "cfo_fy2", "cfo_fy1", "cfo_fy0",
         "net_worth_fy0", "total_debt_fy0", "cash_fy0", "revenue_stub", "pat_stub"]
PLAIN = ["price_low", "price_high", "lot_size", "promoter_pre_pct", "promoter_post_pct", "retail_quota_pct",
         "pre_ipo_price", "eps_fy0", "ronw_fy0_pct", "nav_per_share_fy0"]


def flatten(x: RHPExtract) -> dict:
    """Converted values (₹ crore, shares in crore) plus rhp_page_<field> for each figure.

    Fields whose final value needs the issue price (OFS by seller type, proceeds %, post-issue
    shares) are emitted as helpers and finished in features.derive()."""
    f: dict = {"company_rhp": x.company, "sector": x.sector, "exchange": x.exchanges,
               "financials_basis": x.financials_basis, "fy0_label": x.fy0_label,
               "stub_period_months": x.stub_period_months,
               "pre_ipo_placement": None if x.pre_ipo_placement is None else ("yes" if x.pre_ipo_placement else "no"),
               "lead_managers": "; ".join(x.lead_managers) or None}

    def put(name, value, fig):
        f[name] = value
        if fig is not None and fig.page is not None:
            f[f"rhp_page_{name}"] = fig.page

    for n in MONEY:
        put(n, crore(getattr(x, n)), getattr(x, n))
    for n in PLAIN:
        put(n, plain(getattr(x, n)), getattr(x, n))
    put("fresh_issue_cr", crore(x.fresh_issue_amount), x.fresh_issue_amount)
    put("ofs_cr", crore(x.ofs_amount), x.ofs_amount)
    put("fresh_issue_shares_cr", _shares_cr(x.fresh_issue_shares), x.fresh_issue_shares)
    put("ofs_shares_cr", _shares_cr(x.ofs_shares), x.ofs_shares)
    put("pre_issue_shares_cr", _shares_cr(x.pre_issue_shares), x.pre_issue_shares)
    put("post_issue_shares_cr", _shares_cr(x.post_issue_shares), x.post_issue_shares)

    prom = [_shares_cr(s.shares_offered) for s in x.selling_shareholders if s.kind in ("promoter", "promoter_group")]
    inv = [_shares_cr(s.shares_offered) for s in x.selling_shareholders if s.kind == "investor"]
    # a split is only known if every seller's share count was read; otherwise leave it blank, never 0
    all_known = bool(x.selling_shareholders) and all(_shares_cr(s.shares_offered) is not None
                                                     for s in x.selling_shareholders)
    f["ofs_promoter_shares_cr"] = sum(prom) if all_known else None
    f["ofs_investor_shares_cr"] = sum(inv) if all_known else None

    by_cat: dict[str, float] = {}
    for p in x.objects_of_offer:
        v = crore(p.amount)
        if v is not None:
            by_cat[p.category] = by_cat.get(p.category, 0.0) + v
    for cat in ("debt_repayment", "capex", "gcp"):
        f[f"proceeds_{cat}_cr"] = by_cat.get(cat, 0.0) if x.objects_of_offer else None

    if x.no_listed_peers:
        f.update(peers=None, n_peers=0, peer_pe_list=[], peer_ronw_list=[])
    else:
        f["peers"] = "; ".join(p.nse_symbol or p.name for p in x.peers) or None
        f["n_peers"] = len(x.peers)
        f["peer_pe_list"] = [p.pe for p in x.peers if p.pe is not None and p.pe > 0]
        f["peer_ronw_list"] = [p.ronw_pct for p in x.peers if p.ronw_pct is not None]
        pages = [p.page for p in x.peers if p.page]
        if pages:
            f["rhp_page_peers"] = pages[0]
    return f


def _shares_cr(fig: Fig | None) -> float | None:
    a = absolute(fig)
    return None if a is None else a / 1e7


def checks(f: dict) -> list[str]:
    """Reconciliation checks. Each returned string is a problem."""
    out = []
    g = f.get
    for y in ("fy2", "fy1", "fy0"):
        r, p = g(f"revenue_{y}"), g(f"pat_{y}")
        if r is not None and p is not None and r > 0 and abs(p) > r:
            out.append(f"|PAT| > revenue in {y}: unit mix-up?")
    r0, r1 = g("revenue_fy0"), g("revenue_fy1")
    if r0 and r1 and (r0 / r1 > 50 or r1 / r0 > 50):
        out.append("revenue changes >50x between years: unit mix-up?")
    if g("promoter_pre_pct") is not None and not 0 <= g("promoter_pre_pct") <= 100:
        out.append("promoter_pre_pct outside 0-100")
    if g("promoter_pre_pct") is not None and g("promoter_post_pct") is not None and g("promoter_post_pct") > g("promoter_pre_pct") + 0.5:
        out.append("promoter holding rises after the issue")
    if g("eps_fy0") and g("pat_fy0") and g("pre_issue_shares_cr"):
        implied = g("pat_fy0") / g("pre_issue_shares_cr")          # ₹ crore / crore shares = ₹/share
        if not 0.33 < implied / g("eps_fy0") < 3:
            out.append(f"EPS {g('eps_fy0'):.2f} vs PAT/shares {implied:.2f} disagree (bonus/split after FY end?)")
    if g("retail_quota_pct") not in (None, 10, 35):
        out.append(f"retail quota {g('retail_quota_pct')} is not 10 or 35")
    if g("revenue_fy0") is None or g("pat_fy0") is None:
        out.append("missing fy0 revenue or PAT")
    return out


# --------------------------------------------------------------------------- #
# Claude call
# --------------------------------------------------------------------------- #
class ExtractionError(RuntimeError):
    pass


def find_claude_cli() -> str | None:
    """Claude Code CLI: $CLAUDE_CLI, `claude` on PATH, or the binary bundled with the VS Code extension."""
    import shutil
    if os.environ.get("CLAUDE_CLI"):
        return os.environ["CLAUDE_CLI"]
    if shutil.which("claude"):
        return shutil.which("claude")
    hits = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude*"),
                  key=lambda p: p.stat().st_mtime)
    return str(hits[-1]) if hits else None


class ClaudeCLI:
    """Runs the extraction through `claude -p` (uses the Claude Code login instead of an API key).

    Tools, MCP servers, settings and slash commands are switched off so the call is a plain
    structured-output completion; output is validated against the same RHPExtract schema."""

    def __init__(self, exe: str | None = None, timeout: int = 900):
        self.exe = exe or find_claude_cli()
        if not self.exe:
            raise ExtractionError("Claude Code CLI not found; set CLAUDE_CLI or ANTHROPIC_API_KEY")
        self.timeout = timeout
        self.schema = json.dumps(RHPExtract.model_json_schema())

    def run(self, prompt: str) -> tuple[RHPExtract, dict]:
        import subprocess
        import tempfile
        cmd = [self.exe, "-p", "--output-format", "json", "--model", MODEL, "--tools", "",
               "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
               "--no-session-persistence", "--system-prompt", SYSTEM, "--json-schema", self.schema]
        with tempfile.TemporaryDirectory() as cwd:          # empty cwd: no project CLAUDE.md
            try:
                p = subprocess.run(cmd, input=prompt + "\n\nExtract the fields from these RHP pages.",
                                   capture_output=True, text=True, encoding="utf-8", timeout=self.timeout, cwd=cwd)
            except subprocess.TimeoutExpired:
                raise ExtractionError("claude CLI timed out")
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise ExtractionError(f"claude CLI gave no JSON (exit {p.returncode}): {(p.stderr or p.stdout)[:300]}")
        if out.get("is_error") or out.get("structured_output") is None:
            msg = out.get("result") or out.get("api_error_status") or out.get("subtype")
            if "limit" in str(msg).lower():
                raise RateLimited(str(msg))
            raise ExtractionError(f"claude CLI error: {str(msg)[:300]}")
        x = RHPExtract.model_validate(out["structured_output"])
        u = out.get("usage", {})
        return x, {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                   + u.get("cache_read_input_tokens", 0),
                   "output_tokens": u.get("output_tokens", 0), "backend": "claude-cli",
                   "cost_usd_list": out.get("total_cost_usd"), "session_id": out.get("session_id")}


class RateLimited(RuntimeError):
    pass


def make_client():
    """API client when ANTHROPIC_API_KEY is set, otherwise the Claude Code CLI login."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()
    return ClaudeCLI()


def call_claude(prompt: str, client=None) -> tuple[RHPExtract, dict]:
    if isinstance(client, ClaudeCLI):
        return client.run(prompt)
    import anthropic
    client = client or anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt + "\n\nExtract the fields from these RHP pages."}],
        output_format=RHPExtract,
    )
    if resp.stop_reason == "refusal":
        raise ExtractionError(f"model declined: {getattr(resp.stop_details, 'category', None)}")
    if resp.stop_reason == "max_tokens":
        raise ExtractionError("output hit max_tokens")
    if resp.parsed_output is None:
        raise ExtractionError("no parsed output")
    u = resp.usage
    return resp.parsed_output, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                                "request_id": getattr(resp, "_request_id", None)}


def extract(symbol: str, pdf: Path | None = None, force: bool = False, client=None) -> dict:
    """Run the whole pipeline for one IPO and write data/raw/<symbol>.json."""
    pdf = pdf or RHP_DIR / f"{symbol}.pdf"
    out = RAW_DIR / f"{symbol}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    if not pdf.exists():
        raise ExtractionError(f"no RHP at {pdf}")
    pages = page_texts(pdf)
    if sum(len(p) for p in pages) / max(len(pages), 1) < 500:
        raise ExtractionError("PDF has almost no text layer (scanned). OCR it first, e.g. `ocrmypdf in.pdf out.pdf`.")
    idx = select_pages(pages)
    x, usage = call_claude(build_prompt(pages, idx), client)
    flat = flatten(x)
    rec = {"symbol": symbol, "rhp_file": pdf.name, "model": MODEL,
           "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "pages_sent": [i + 1 for i in idx], "n_pdf_pages": len(pages), "usage": usage,
           "checks": checks(flat), "fields": flat, "extract": x.model_dump()}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def peer_stats(pe_list: list[float], ronw_list: list[float]) -> dict:
    return {"peer_pe_median": median(pe_list) if pe_list else None,
            "peer_pe_min": min(pe_list) if pe_list else None,
            "peer_pe_max": max(pe_list) if pe_list else None,
            "peer_ronw_median_pct": median(ronw_list) if ronw_list else None}
