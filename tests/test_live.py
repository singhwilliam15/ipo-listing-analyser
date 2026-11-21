import numpy as np
import pandas as pd

from src import live


def _hist(n=200, seed=0):
    r = np.random.default_rng(seed)
    sub = np.exp(r.normal(3, 1.5, n))
    gain = np.where(sub > 10, r.normal(25, 15, n), r.normal(-3, 10, n))
    return pd.DataFrame({"sub_total_x": sub, "listing_gain_pct": gain, "loss_making_flag": r.integers(0, 2, n),
                         "debt_to_equity_fy0": r.uniform(0, 3, n), "revenue_cagr_pct": r.uniform(-5, 60, n),
                         "ronw_fy0_pct": r.uniform(-10, 40, n), "pat_margin_fy0_pct": r.uniform(-5, 20, n),
                         "pe_premium_to_peers_pct": r.uniform(-50, 100, n), "proceeds_debt_repayment_pct": r.uniform(0, 100, n),
                         "ipo_premium_to_preipo_pct": r.uniform(-20, 40, n)})


def test_subscription_table_is_monotone_and_complete():
    t = live.subscription_table(_hist())
    assert list(t.index) == live.SUB_LABELS
    assert t["gain_rate"].iloc[-1] > t["gain_rate"].iloc[0]


def test_flags_mark_leverage_and_losses_bad_with_evidence():
    row = pd.Series({"loss_making_flag": 1, "debt_to_equity_fy0": 3.0, "revenue_cagr_pct": 40.0, "ofs_pct": 100.0})
    fl = live.flags(row, _hist())
    kinds = {f.text.split(":")[0]: f.kind for f in fl}
    assert kinds["High leverage"] == "bad"
    assert any(f.kind == "bad" and f.text.startswith("Loss-making") and "similar IPOs" in f.evidence for f in fl)
    assert any(f.kind == "good" and f.text.startswith("Revenue CAGR") for f in fl)
    assert any(f.kind == "info" and "offer for sale" in f.text for f in fl)


def test_flags_ignore_missing_values():
    assert live.flags(pd.Series({"debt_to_equity_fy0": np.nan}), _hist()) == []


def test_find_issue_fuzzy_match(monkeypatch):
    fake = pd.DataFrame([{"symbol": "MAHACOAL", "company": "Mahanadi Coalfields Limited", "status": "Active",
                          "segment": "Mainboard", "price_low": 129.0, "price_high": 136.0,
                          "issue_open": "19-Nov-2026", "issue_close": "23-Nov-2026"}])
    monkeypatch.setattr(live, "list_open_issues", lambda nse=None: fake)
    m = live.find_issue("mahanadi coal", include_past=False)
    assert m and m[0].symbol == "MAHACOAL"
    assert live.find_issue("zzzzzz", include_past=False) == []


def test_date_parsing_handles_nse_and_iso():
    assert live._date("23-Nov-2026") == pd.Timestamp("2026-11-23")
    assert live._date("2026-03-27") == pd.Timestamp("2026-03-27")
    assert pd.isna(live._date(None))
