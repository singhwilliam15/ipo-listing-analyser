"""Column schema for data/ipos.csv (one row per IPO).

Every column is declared once here. Order in this list is the column order in the CSV.
`source` says where a value comes from:
    nse      NSE issue list / bid data
    rhp      extracted from the RHP by src/extract_rhp.py (page refs in data/raw/<symbol>.json)
    derived  computed in src/features.py, never hand-entered
    market   public price data as of issue_close
    manual   typed in via data/manual/overrides.csv (no public historical source)
    outcome  listing-day results; NEVER used as a model feature
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str      # text | date | number | integer | yesno | flag
    group: str      # A..G as in the brief, Q = bookkeeping
    source: str
    note: str = ""


C = Column
COLUMNS: list[Column] = [
    # A. Identity
    C("symbol", "text", "A", "nse"),
    C("company", "text", "A", "nse"),
    C("sector", "text", "A", "rhp", "one of SECTORS"),
    C("issue_open", "date", "A", "nse"),
    C("issue_close", "date", "A", "nse"),
    C("listing_date", "date", "A", "nse"),
    C("exchange", "text", "A", "rhp", "NSE / BSE / both"),
    # B. Issue structure
    C("price_low", "number", "B", "nse", "band floor"),
    C("price_high", "number", "B", "nse", "band cap; issue price if fixed"),
    C("issue_price", "number", "B", "nse", "final price"),
    C("lot_size", "integer", "B", "nse"),
    C("fresh_issue_cr", "number", "B", "rhp"),
    C("ofs_cr", "number", "B", "rhp"),
    C("issue_size_cr", "number", "B", "derived", "fresh + OFS"),
    C("ofs_pct", "number", "B", "derived"),
    C("ofs_by_promoter_cr", "number", "B", "rhp"),
    C("ofs_by_investors_cr", "number", "B", "rhp"),
    C("promoter_pre_pct", "number", "B", "rhp"),
    C("promoter_post_pct", "number", "B", "rhp"),
    C("promoter_dilution_pp", "number", "B", "derived"),
    C("post_issue_shares_cr", "number", "B", "rhp"),
    C("mcap_at_issue_cr", "number", "B", "derived"),
    C("proceeds_debt_repayment_pct", "number", "B", "rhp"),
    C("proceeds_capex_pct", "number", "B", "rhp"),
    C("proceeds_gcp_pct", "number", "B", "rhp"),
    C("anchor_cr", "number", "B", "manual", "anchor book; from exchange notice"),
    C("anchor_pct_of_issue", "number", "B", "derived"),
    C("retail_quota_pct", "number", "B", "rhp", "35 or 10"),
    C("pre_ipo_placement", "yesno", "B", "rhp"),
    C("pre_ipo_price", "number", "B", "rhp"),
    C("ipo_premium_to_preipo_pct", "number", "B", "derived"),
    C("lead_managers", "text", "B", "rhp", "semicolon-separated"),
    C("n_lead_managers", "integer", "B", "derived"),
    # C. Financials (restated consolidated, ₹ crore)
    *[C(f"{m}_{y}", "number", "C", "rhp") for m in ("revenue", "ebitda", "pat", "cfo") for y in ("fy2", "fy1", "fy0")],
    C("net_worth_fy0", "number", "C", "rhp"),
    C("total_debt_fy0", "number", "C", "rhp"),
    C("cash_fy0", "number", "C", "rhp"),
    C("eps_fy0", "number", "C", "rhp"),
    C("ronw_fy0_pct", "number", "C", "rhp"),
    C("nav_per_share_fy0", "number", "C", "rhp"),
    C("stub_period_months", "integer", "C", "rhp"),
    C("revenue_stub", "number", "C", "rhp"),
    C("pat_stub", "number", "C", "rhp"),
    C("revenue_cagr_pct", "number", "C", "derived"),
    C("pat_cagr_pct", "number", "C", "derived"),
    C("ebitda_margin_fy0_pct", "number", "C", "derived"),
    C("pat_margin_fy0_pct", "number", "C", "derived"),
    C("pat_margin_change_pp", "number", "C", "derived"),
    C("pat_growth_fy0_pct", "number", "C", "derived"),
    C("pat_jump_flag", "flag", "C", "derived"),
    C("cfo_to_pat", "number", "C", "derived"),
    C("debt_to_equity_fy0", "number", "C", "derived"),
    C("pe_at_issue", "number", "C", "derived"),
    C("pb_at_issue", "number", "C", "derived"),
    C("ps_at_issue", "number", "C", "derived"),
    C("loss_making_flag", "flag", "C", "derived"),
    # D. Peers (as printed in the RHP)
    C("peers", "text", "D", "rhp"),
    C("n_peers", "integer", "D", "derived"),
    C("peer_pe_median", "number", "D", "derived"),
    C("peer_pe_min", "number", "D", "derived"),
    C("peer_pe_max", "number", "D", "derived"),
    C("pe_premium_to_peers_pct", "number", "D", "derived"),
    C("peer_ronw_median_pct", "number", "D", "derived"),
    # E. Demand (final bid data)
    C("sub_qib_x", "number", "E", "nse"),
    C("sub_nii_x", "number", "E", "nse"),
    C("sub_bnii_x", "number", "E", "nse"),
    C("sub_snii_x", "number", "E", "nse"),
    C("sub_retail_x", "number", "E", "nse"),
    C("sub_employee_x", "number", "E", "nse"),
    C("sub_total_x", "number", "E", "nse"),
    C("n_applications", "integer", "E", "nse"),
    C("gmp_last_rs", "number", "E", "manual"),
    C("gmp_last_pct", "number", "E", "derived"),
    # F. Market context (as of issue_close)
    C("nifty_ret_1m_pct", "number", "F", "market"),
    C("nifty_ret_3m_pct", "number", "F", "market"),
    C("india_vix_close", "number", "F", "market"),
    C("ipos_listed_prev_30d", "integer", "F", "derived"),
    C("avg_listing_gain_prev_5_pct", "number", "F", "derived"),
    # G. Outcomes (never features)
    C("list_open", "number", "G", "outcome", "split-unadjusted"),
    C("list_high", "number", "G", "outcome"),
    C("list_low", "number", "G", "outcome"),
    C("list_close", "number", "G", "outcome"),
    C("listing_gain_pct", "number", "G", "outcome", "primary target"),
    C("day1_close_gain_pct", "number", "G", "outcome"),
    C("target_gain", "flag", "G", "outcome"),
    C("target_gain_5", "flag", "G", "outcome"),
    C("ret_1w_pct", "number", "G", "outcome"),
    C("ret_1m_pct", "number", "G", "outcome"),
    # Bookkeeping
    C("data_quality", "text", "Q", "derived", "full / partial / missing_rhp"),
]
del C

ORDER = [c.name for c in COLUMNS]
BY_NAME = {c.name: c for c in COLUMNS}
OUTCOMES = [c.name for c in COLUMNS if c.group == "G"]
# Numeric columns a model may use. Identity, text and outcome columns are excluded.
FEATURES = [c.name for c in COLUMNS
            if c.group in "BCDEF" and c.dtype in ("number", "integer", "yesno", "flag")]
CATEGORICAL_FEATURES = ["sector"]

SECTORS = [
    "Financial services", "Banking & NBFC", "IT & software", "Internet & platforms",
    "Healthcare & pharma", "Consumer & retail", "FMCG & food", "Auto & auto components",
    "Industrials & capital goods", "Electronics & EMS", "Chemicals & materials",
    "Energy, power & renewables", "Infrastructure, construction & real estate",
    "Logistics & travel", "Media, education & services",
]

TRAIN_START = "2022-01-01"
TEST_START = "2026-01-01"


def empty_frame():
    import pandas as pd
    return pd.DataFrame(columns=ORDER)


def conform(df):
    """Return df with exactly the schema columns, in order, with consistent dtypes."""
    import pandas as pd
    out = df.reindex(columns=ORDER).copy()
    for c in COLUMNS:
        s = out[c.name]
        if c.dtype == "date":
            out[c.name] = pd.to_datetime(s, errors="coerce").dt.date
        elif c.dtype in ("number",):
            out[c.name] = pd.to_numeric(s, errors="coerce").astype(float)
        elif c.dtype in ("integer", "flag"):
            out[c.name] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif c.dtype == "yesno":
            out[c.name] = s.map(_yesno).astype(object)
    return out


def _yesno(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        return "yes" if t in ("yes", "y", "true", "1") else "no" if t in ("no", "n", "false", "0") else None
    return "yes" if bool(v) else "no"
