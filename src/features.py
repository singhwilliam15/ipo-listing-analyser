"""Derived fields, market context, targets and data quality. Pure pandas, no I/O.

All derived columns are recomputed every build, so never hand-edit them in ipos.csv;
fix the inputs (data/manual/overrides.csv) instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .extract_rhp import peer_stats


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)


def _div(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a / b.where(b != 0)).astype(float)


def _fill(df: pd.DataFrame, name: str, values: pd.Series) -> None:
    """Fill only missing cells of df[name] (so extracted/overridden values win)."""
    df[name] = _col(df, name).fillna(values)


def cagr(end, start, years: int = 2) -> pd.Series:
    end, start = pd.to_numeric(end, errors="coerce"), pd.to_numeric(start, errors="coerce")
    ok = (end > 0) & (start > 0)
    return ((end / start.where(ok)) ** (1 / years) - 1) * 100


def derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    p = _col(d, "issue_price")

    # --- B. issue structure -------------------------------------------------
    _fill(d, "fresh_issue_cr", _col(d, "fresh_issue_shares_cr") * p)      # crore shares x ₹ = ₹ crore
    _fill(d, "ofs_cr", _col(d, "ofs_shares_cr") * p)
    fresh, ofs = _col(d, "fresh_issue_cr"), _col(d, "ofs_cr")
    any_part = fresh.notna() | ofs.notna()
    d["issue_size_cr"] = (fresh.fillna(0) + ofs.fillna(0)).where(any_part)
    d["ofs_pct"] = _div(ofs.fillna(0).where(any_part), d["issue_size_cr"]) * 100
    _fill(d, "ofs_by_promoter_cr", _col(d, "ofs_promoter_shares_cr") * p)
    _fill(d, "ofs_by_investors_cr", _col(d, "ofs_investor_shares_cr") * p)
    _fill(d, "post_issue_shares_cr", _col(d, "pre_issue_shares_cr") + _div(fresh, p).fillna(0).where(fresh.notna() | ofs.notna()))
    # RHPs print post-issue promoter holding as [●]; derive it from pre-holding, promoter OFS and new shares
    no_ofs = ofs.fillna(0).eq(0) & fresh.notna()                  # fresh-issue only: promoters sell nothing
    promo_ofs = _col(d, "ofs_promoter_shares_cr").where(~no_ofs, 0)  # else unknown stays unknown
    promo_post_shares = _col(d, "promoter_pre_pct") / 100 * _col(d, "pre_issue_shares_cr") - promo_ofs
    _fill(d, "promoter_post_pct", (_div(promo_post_shares, _col(d, "post_issue_shares_cr")) * 100).clip(0, 100))
    d["promoter_dilution_pp"] = _col(d, "promoter_pre_pct") - _col(d, "promoter_post_pct")
    d["mcap_at_issue_cr"] = p * _col(d, "post_issue_shares_cr")
    for cat in ("debt_repayment", "capex", "gcp"):
        _fill(d, f"proceeds_{cat}_pct", _div(_col(d, f"proceeds_{cat}_cr"), fresh) * 100)
    _fill(d, "anchor_cr", _col(d, "anchor_shares_cr") * p)        # anchor portion from the NSE issue page
    d["anchor_pct_of_issue"] = _div(_col(d, "anchor_cr"), d["issue_size_cr"]) * 100
    d["ipo_premium_to_preipo_pct"] = (_div(p, _col(d, "pre_ipo_price")) - 1) * 100
    lm = d["lead_managers"] if "lead_managers" in d else pd.Series(None, index=d.index, dtype=object)
    d["n_lead_managers"] = lm.map(lambda s: len([x for x in str(s).split(";") if x.strip()]) if isinstance(s, str) and s.strip() else np.nan)

    # --- C. financials -------------------------------------------------------
    rev0, rev2 = _col(d, "revenue_fy0"), _col(d, "revenue_fy2")
    pat0, pat1, pat2 = _col(d, "pat_fy0"), _col(d, "pat_fy1"), _col(d, "pat_fy2")
    d["revenue_cagr_pct"] = cagr(rev0, rev2)
    d["pat_cagr_pct"] = cagr(pat0, pat2)
    d["ebitda_margin_fy0_pct"] = _div(_col(d, "ebitda_fy0"), rev0.where(rev0 > 0)) * 100
    d["pat_margin_fy0_pct"] = _div(pat0, rev0.where(rev0 > 0)) * 100
    d["pat_margin_change_pp"] = d["pat_margin_fy0_pct"] - _div(pat2, rev2.where(rev2 > 0)) * 100
    g0 = (_div(pat0, pat1.where(pat1 > 0)) - 1) * 100
    g1 = (_div(pat1, pat2.where(pat2 > 0)) - 1) * 100
    d["pat_growth_fy0_pct"] = g0
    # Jump: >50% growth that is also >3x last year's growth (or last year's growth was flat/negative/unknown)
    jump = (g0 > 50) & (g1.isna() | (g1 <= 0) | (g0 > 3 * g1))
    d["pat_jump_flag"] = jump.astype("Int64").where(g0.notna())
    cfo = sum(_col(d, f"cfo_{y}") for y in ("fy2", "fy1", "fy0"))
    pats = pat0 + pat1 + pat2
    d["cfo_to_pat"] = _div(cfo, pats.where(pats > 0))
    nw = _col(d, "net_worth_fy0")
    d["debt_to_equity_fy0"] = _div(_col(d, "total_debt_fy0"), nw.where(nw > 0))
    d["pe_at_issue"] = _div(d["mcap_at_issue_cr"], pat0.where(pat0 > 0))
    nav = _col(d, "nav_per_share_fy0")
    d["pb_at_issue"] = _div(p, nav.where(nav > 0))
    d["ps_at_issue"] = _div(d["mcap_at_issue_cr"], rev0.where(rev0 > 0))
    d["loss_making_flag"] = (pat0 <= 0).astype("Int64").where(pat0.notna())

    # --- D. peers (lists come from the raw RHP JSON) -------------------------
    if "peer_pe_list" in d:
        stats = [peer_stats(_list(a), _list(b)) for a, b in zip(d["peer_pe_list"], d.get("peer_ronw_list", [None] * len(d)))]
        for k in ("peer_pe_median", "peer_pe_min", "peer_pe_max", "peer_ronw_median_pct"):
            _fill(d, k, pd.Series([s[k] for s in stats], index=d.index, dtype=float))
    d["pe_premium_to_peers_pct"] = (_div(d["pe_at_issue"], _col(d, "peer_pe_median")) - 1) * 100

    # --- E. demand -----------------------------------------------------------
    d["gmp_last_pct"] = _div(_col(d, "gmp_last_rs"), p) * 100

    # --- G. outcomes ---------------------------------------------------------
    d["listing_gain_pct"] = (_div(_col(d, "list_open"), p) - 1) * 100
    d["day1_close_gain_pct"] = (_div(_col(d, "list_close"), p) - 1) * 100
    lg = d["listing_gain_pct"]
    d["target_gain"] = (lg > 0).astype("Int64").where(lg.notna())
    d["target_gain_5"] = (lg > 5).astype("Int64").where(lg.notna())
    return d


def _list(v) -> list[float]:
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v if x is not None]
    return []


def market_context(d: pd.DataFrame, market: pd.DataFrame | None, history: pd.DataFrame) -> pd.DataFrame:
    """Section F, all as of each row's issue_close.

    market:  index = date, columns nifty, vix (daily closes)
    history: every mainboard IPO incl. warm-up rows before the training window,
             columns listing_date, listing_gain_pct. Only listings strictly before
             issue_close are counted, so nothing from the future leaks in."""
    d = d.copy()
    close = pd.to_datetime(d["issue_close"], errors="coerce")
    if market is not None and not market.empty:
        m = market.sort_index()
        m.index = pd.to_datetime(m.index)

        def asof(col, when):
            return np.array([m[col].asof(t) if pd.notna(t) and t >= m.index[0] else np.nan for t in when], dtype=float)

        n0 = asof("nifty", close)
        d["nifty_ret_1m_pct"] = (n0 / asof("nifty", close - pd.Timedelta(days=30)) - 1) * 100
        d["nifty_ret_3m_pct"] = (n0 / asof("nifty", close - pd.Timedelta(days=91)) - 1) * 100
        d["india_vix_close"] = asof("vix", close)
    h = history.assign(listing_date=pd.to_datetime(history["listing_date"], errors="coerce"),
                       listing_gain_pct=pd.to_numeric(history["listing_gain_pct"], errors="coerce"))
    h = h.dropna(subset=["listing_date"]).sort_values("listing_date")
    prev30, prev5 = [], []
    for t in close:
        if pd.isna(t):
            prev30.append(np.nan)
            prev5.append(np.nan)
            continue
        before = h[h["listing_date"] < t]
        prev30.append(int((before["listing_date"] >= t - pd.Timedelta(days=30)).sum()))
        g = before["listing_gain_pct"].dropna().tail(5)
        prev5.append(float(g.mean()) if len(g) == 5 else np.nan)
    d["ipos_listed_prev_30d"] = prev30
    d["avg_listing_gain_prev_5_pct"] = prev5
    return d


KEY_FIELDS = ["revenue_fy0", "pat_fy0", "issue_size_cr", "post_issue_shares_cr", "sub_total_x", "listing_gain_pct"]


def data_quality(d: pd.DataFrame) -> pd.Series:
    has_rhp = d["has_rhp"].fillna(False).astype(bool) if "has_rhp" in d else pd.Series(False, index=d.index)
    n_checks = pd.to_numeric(d["rhp_check_count"], errors="coerce").fillna(0) if "rhp_check_count" in d else 0
    complete = d.reindex(columns=KEY_FIELDS).notna().all(axis=1)
    q = np.where(~has_rhp, "missing_rhp", np.where(complete & (n_checks == 0), "full", "partial"))
    return pd.Series(q, index=d.index)
