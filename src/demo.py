"""Synthetic IPO rows with the real schema. For tests and for trying the app before the
real dataset exists. Every company here is fake and the relationships are invented."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, schema


def synthetic(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    listing = pd.to_datetime("2021-11-01") + pd.to_timedelta(np.sort(rng.integers(0, 365 * 5, n)), unit="D")
    close = listing - pd.to_timedelta(rng.integers(3, 7, n), unit="D")
    price = rng.choice([90, 150, 275, 420, 600, 1100], n) * rng.uniform(0.8, 1.2, n)
    rev0 = np.exp(rng.normal(6.5, 1.2, n))
    growth = rng.normal(0.2, 0.25, n)
    margin = rng.normal(0.09, 0.07, n)
    shares = rev0 * rng.uniform(1.5, 6, n) / price
    ofs_share = rng.uniform(0, 1, n)
    issue = shares * price * rng.uniform(0.1, 0.3, n)
    sub = np.exp(rng.normal(2.5, 1.6, n))
    df = pd.DataFrame({
        "symbol": [f"DEMO{i:03d}" for i in range(n)], "company": [f"Demo Company {i}" for i in range(n)],
        "sector": rng.choice(schema.SECTORS, n), "issue_open": close - pd.Timedelta(days=2),
        "issue_close": close, "listing_date": listing, "exchange": "both",
        "price_low": price * 0.95, "price_high": price, "issue_price": price, "lot_size": np.round(14000 / price),
        "fresh_issue_cr": issue * (1 - ofs_share), "ofs_cr": issue * ofs_share,
        "ofs_promoter_shares_cr": issue * ofs_share * rng.uniform(0, 1, n) / price,
        "promoter_pre_pct": rng.uniform(40, 95, n),
        "post_issue_shares_cr": shares, "retail_quota_pct": rng.choice([35, 10], n, p=[0.6, 0.4]),
        "pre_ipo_placement": rng.choice(["yes", "no"], n, p=[0.3, 0.7]), "lead_managers": "A; B; C",
        "revenue_fy2": rev0 / (1 + growth) ** 2, "revenue_fy1": rev0 / (1 + growth), "revenue_fy0": rev0,
        "ebitda_fy0": rev0 * (margin + 0.06), "pat_fy2": rev0 / (1 + growth) ** 2 * margin * 0.8,
        "pat_fy1": rev0 / (1 + growth) * margin * 0.9, "pat_fy0": rev0 * margin,
        "cfo_fy2": rev0 * margin * rng.uniform(0, 1.3, n), "cfo_fy1": rev0 * margin * rng.uniform(0, 1.3, n),
        "cfo_fy0": rev0 * margin * rng.uniform(0, 1.3, n), "net_worth_fy0": rev0 * rng.uniform(0.3, 1.2, n),
        "total_debt_fy0": rev0 * rng.uniform(0, 0.8, n), "nav_per_share_fy0": price / rng.uniform(2, 12, n),
        "peer_pe_median": rng.uniform(20, 70, n),
        "sub_qib_x": sub * rng.uniform(0.8, 2.5, n), "sub_nii_x": sub * rng.uniform(0.5, 2, n),
        "sub_retail_x": sub * rng.uniform(0.2, 0.8, n), "sub_total_x": sub,
    })
    df["promoter_post_pct"] = df["promoter_pre_pct"] * rng.uniform(0.7, 0.95, n)
    d = features.derive(df)
    # Invented "truth": demand dominates, valuation and OFS-heavy issues drag
    lg = (6 * np.log(d["sub_total_x"]) - 8 - 0.08 * d["ofs_pct"].fillna(50)
          - 0.05 * d["pe_premium_to_peers_pct"].clip(-50, 150).fillna(0) + rng.normal(0, 12, n))
    d["list_open"] = price * (1 + lg / 100)
    d["list_close"] = d["list_open"] * (1 + rng.normal(0, 0.05, n))
    d["list_high"] = d[["list_open", "list_close"]].max(axis=1) * 1.03
    d["list_low"] = d[["list_open", "list_close"]].min(axis=1) * 0.97
    d = features.derive(d)
    market = pd.DataFrame({"nifty": 17000 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, 2200))),
                           "vix": np.clip(14 + np.cumsum(rng.normal(0, 0.4, 2200)) * 0.1, 9, 30)},
                          index=pd.bdate_range("2021-06-01", periods=2200))
    d = features.market_context(d, market, d[["listing_date", "listing_gain_pct"]])
    d["has_rhp"], d["rhp_check_count"] = True, 0
    d["data_quality"] = features.data_quality(d)
    d = d[d["listing_date"] >= pd.Timestamp(schema.TRAIN_START)]
    return schema.conform(d).reset_index(drop=True)
