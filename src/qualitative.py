"""Second Claude pass over an RHP: business, industry, promoters, risks, litigation.

Same rule as the numbers: Claude only reads. It reports what the document says, with the PDF page
for every point, and any figure is copied as printed with its unit. Nothing is calculated here.
Cached in data/raw/<SYMBOL>.qual.json and dropped when the RHP file changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import extract_rhp as X

# (name, heading regex matched near the top of a page, pages to take from there)
QUAL_HEADINGS: list[tuple[str, str, int]] = [
    ("risk_factors", r"\bRISK FACTORS\b", 10),
    ("business", r"\bOUR BUSINESS\b", 8),
    ("industry", r"\bINDUSTRY OVERVIEW\b", 4),
    ("promoters", r"\bOUR PROMOTERS?( AND PROMOTER GROUP)?\b", 3),
    ("litigation", r"OUTSTANDING LITIGATION", 4),
]
MAX_QUAL_PAGES = 34


class Point(BaseModel):
    text: str = Field(description="One short factual sentence taken from the pages. No opinion and no calculation.")
    page: Optional[int] = Field(description="Number from the '=== PAGE n ===' marker where this is stated.")


class Segment(BaseModel):
    name: str
    share_of_revenue: Optional[str] = Field(description="Share of revenue exactly as printed (e.g. '62.4%'); null if not printed.")
    page: Optional[int]


class Concentration(BaseModel):
    text: str = Field(description="Customer, supplier, product or geography concentration as printed, e.g. 'top 5 customers contributed 71% of revenue in FY24'.")
    page: Optional[int]


class Promoter(BaseModel):
    name: str
    background: str = Field(description="Role and experience as printed.")
    page: Optional[int]


class Risk(BaseModel):
    title: str
    detail: str = Field(description="What the RHP says the risk is, in one or two sentences, with any figure as printed.")
    severity: Literal["high", "medium", "low"] = Field(
        description="high = could materially impair revenue, licences, solvency or listing value and is specific to this company; low = generic.")
    page: Optional[int]


class LitigationItem(BaseModel):
    category: str = Field(description="e.g. criminal, tax, regulatory, civil, against promoters, against directors, material litigation.")
    count: Optional[str] = Field(description="Number of cases as printed; null if not given.")
    amount: Optional[str] = Field(description="Amount involved as printed with its unit; null if not given.")
    page: Optional[int]


class QualExtract(BaseModel):
    business_summary: list[Point] = Field(description="3-5 points: what the company does, how it earns, scale, footprint.")
    segments: list[Segment]
    concentration: list[Concentration]
    industry: list[Point] = Field(description="3-5 points on industry size, growth and the company's position, as stated in the RHP (usually a commissioned report).")
    promoters: list[Promoter]
    strengths: list[Point] = Field(description="Strengths the company itself lists, 4-6.")
    risks: list[Risk] = Field(description="The 6-8 most company-specific risks from Risk Factors, not generic market risks.")
    litigation: list[LitigationItem]
    contingent_liabilities: list[Point]
    related_party: list[Point] = Field(description="Related-party transactions or arrangements of note, as stated.")
    red_flags: list[Point] = Field(description="Facts that an analyst would query: qualified audit opinions or emphasis of matter, promoter pledges, "
                                               "regulatory actions, delays in filings, large related-party dependence, revenue concentration. Only if stated in these pages.")
    notes: list[str]


SYSTEM = """You read selected sections of the Red Herring Prospectus (RHP) of an Indian IPO and report what it says.
Each page starts with a marker '=== PAGE n ==='.

Rules:
- Report only what the pages state. Do not use outside knowledge, do not opine and do not calculate.
- Every point carries page = the n of the marker of the page it came from.
- Copy figures exactly as printed with their unit. Use null when a figure is not on the pages.
- Prefer facts specific to this company. Skip boilerplate risks (market volatility, general economy).
- If a section is missing from the pages, return an empty list for it and say so in notes."""


def select_qual_pages(pages: list[str]) -> list[int]:
    chosen: list[int] = []
    for name, pat, span in QUAL_HEADINGS:
        i = X.find_heading(pages, pat, start=3)
        if i is not None:
            chosen += list(range(i, min(i + span, len(pages))))
    seen: list[int] = []
    for i in chosen:
        if i not in seen and pages[i].strip():
            seen.append(i)
    return sorted(seen[:MAX_QUAL_PAGES])


def qual_path(symbol: str) -> Path:
    return X.RAW_DIR / f"{symbol}.qual.json"


def extract_qual(symbol: str, pdf: Path | None = None, force: bool = False, client=None) -> dict:
    """Run (or load) the qualitative pass for one IPO."""
    pdf = pdf or X.RHP_DIR / f"{symbol}.pdf"
    out = qual_path(symbol)
    sha = X.file_sha256(pdf) if pdf.exists() else None
    if out.exists() and not force:
        rec = json.loads(out.read_text(encoding="utf-8"))
        if not (sha and rec.get("rhp_sha256") and rec["rhp_sha256"] != sha):
            return rec
        out.unlink()
    if not pdf.exists():
        raise X.ExtractionError(f"no RHP at {pdf}")
    pages = X.page_texts(pdf)
    idx = select_qual_pages(pages)
    if not idx:
        raise X.ExtractionError("could not find Risk Factors / Our Business sections (scanned or unusual layout)")
    x, usage = X.call_claude(X.build_prompt(pages, idx), client or X.make_client(), QualExtract, SYSTEM)
    rec = {"symbol": symbol, "rhp_sha256": sha, "model": X.MODEL, "pages_sent": [i + 1 for i in idx], "usage": usage,
           "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "qual": x.model_dump()}
    X.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec
