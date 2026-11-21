import numpy as np
import pandas as pd
import pytest

from src import features


def row(**kw):
    base = dict(symbol="X", issue_price=100.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_issue_structure_from_shares():
    d = features.derive(row(fresh_issue_shares_cr=0.5, ofs_shares_cr=1.5, pre_issue_shares_cr=4.0,
                            ofs_promoter_shares_cr=1.0, ofs_investor_shares_cr=0.5))
    r = d.iloc[0]
    assert r.fresh_issue_cr == pytest.approx(50)        # 0.5 crore shares x ₹100
    assert r.ofs_cr == pytest.approx(150)
    assert r.issue_size_cr == pytest.approx(200)
    assert r.ofs_pct == pytest.approx(75)
    assert r.ofs_by_promoter_cr == pytest.approx(100)
    assert r.post_issue_shares_cr == pytest.approx(4.5)  # pre + fresh/price
    assert r.mcap_at_issue_cr == pytest.approx(450)


def test_extracted_amount_wins_over_shares():
    d = features.derive(row(fresh_issue_cr=80.0, fresh_issue_shares_cr=0.5))
    assert d.iloc[0].fresh_issue_cr == 80


def test_missing_stays_missing_not_zero():
    d = features.derive(row())
    r = d.iloc[0]
    assert np.isnan(r.issue_size_cr) and np.isnan(r.ofs_pct) and np.isnan(r.pe_at_issue)
    assert pd.isna(r.pat_jump_flag) and pd.isna(r.loss_making_flag) and pd.isna(r.target_gain)


def test_financial_ratios():
    d = features.derive(row(post_issue_shares_cr=10.0, revenue_fy2=100.0, revenue_fy1=121.0, revenue_fy0=144.0,
                            pat_fy2=10.0, pat_fy1=12.0, pat_fy0=20.0, ebitda_fy0=28.8,
                            cfo_fy2=5.0, cfo_fy1=10.0, cfo_fy0=15.0, net_worth_fy0=200.0, total_debt_fy0=50.0,
                            nav_per_share_fy0=25.0, peer_pe_median=40.0))
    r = d.iloc[0]
    assert r.revenue_cagr_pct == pytest.approx(20)
    assert r.ebitda_margin_fy0_pct == pytest.approx(20)
    assert r.pat_margin_change_pp == pytest.approx(20 / 144 * 100 - 10)
    assert r.pat_growth_fy0_pct == pytest.approx(66.667, rel=1e-3)
    assert r.pat_jump_flag == 1                           # 66.7% > 50% and > 3 x 20%
    assert r.cfo_to_pat == pytest.approx(30 / 42)
    assert r.debt_to_equity_fy0 == pytest.approx(0.25)
    assert r.pe_at_issue == pytest.approx(1000 / 20)     # mcap 1000 / PAT 20
    assert r.pb_at_issue == pytest.approx(4)
    assert r.ps_at_issue == pytest.approx(1000 / 144)
    assert r.pe_premium_to_peers_pct == pytest.approx(25)
    assert r.loss_making_flag == 0


def test_no_jump_when_growth_is_steady():
    d = features.derive(row(pat_fy2=10.0, pat_fy1=16.0, pat_fy0=25.6))   # 60% then 60%
    assert d.iloc[0].pat_jump_flag == 0


def test_loss_maker_has_no_pe():
    d = features.derive(row(post_issue_shares_cr=10.0, pat_fy0=-5.0, revenue_fy0=50.0))
    assert d.iloc[0].loss_making_flag == 1 and np.isnan(d.iloc[0].pe_at_issue)


def test_targets():
    d = features.derive(pd.DataFrame({"symbol": list("abc"), "issue_price": [100.0] * 3,
                                      "list_open": [104.0, 99.0, np.nan], "list_close": [110.0, 95.0, np.nan]}))
    assert list(d["listing_gain_pct"].round(6)[:2]) == [4.0, -1.0]
    assert d["target_gain"].tolist()[:2] == [1, 0] and pd.isna(d["target_gain"].iloc[2])
    assert d["target_gain_5"].tolist()[:2] == [0, 0]
    assert d["day1_close_gain_pct"].iloc[0] == pytest.approx(10)


def test_peer_lists():
    df = row(peer_pe_list=[30.0, 50.0, 40.0], peer_ronw_list=[10.0, 20.0])
    r = features.derive(df).iloc[0]
    assert (r.peer_pe_median, r.peer_pe_min, r.peer_pe_max, r.peer_ronw_median_pct) == (40, 30, 50, 15)


def test_market_context_uses_only_the_past():
    market = pd.DataFrame({"nifty": np.linspace(100, 200, 200), "vix": 15.0},
                          index=pd.date_range("2023-01-01", periods=200))
    hist = pd.DataFrame({"listing_date": pd.to_datetime(["2023-03-01", "2023-03-05", "2023-03-10", "2023-03-15",
                                                         "2023-03-20", "2023-04-10"]),
                         "listing_gain_pct": [10, 20, 30, 40, 50, 999]})
    d = pd.DataFrame({"issue_close": pd.to_datetime(["2023-04-01"])})
    out = features.market_context(d, market, hist).iloc[0]
    assert out.avg_listing_gain_prev_5_pct == 30          # the 999 listed later and must not count
    assert out.ipos_listed_prev_30d == 4                  # 1 Mar is more than 30 days before 1 Apr
    assert out.india_vix_close == 15
    assert out.nifty_ret_1m_pct > 0


def test_data_quality():
    d = pd.DataFrame({"has_rhp": [True, True, None], "rhp_check_count": [0, 2, None],
                      **{k: [1.0, 1.0, 1.0] for k in features.KEY_FIELDS}})
    assert features.data_quality(d).tolist() == ["full", "partial", "missing_rhp"]


def test_promoter_post_pct_derived_for_pure_ofs():
    """Tata Technologies: TML held 64.79% of 40.567 cr shares and sold 4.6275 cr in a pure OFS -> ~53.4% after."""
    import pandas as pd
    from src import features
    df = pd.DataFrame({"issue_price": [500.0], "ofs_shares_cr": [6.0850278], "pre_issue_shares_cr": [40.566853],
                       "promoter_pre_pct": [64.79], "ofs_promoter_shares_cr": [4.6275]})
    d = features.derive(df)
    assert abs(d.at[0, "promoter_post_pct"] - 53.38) < 0.05
    assert abs(d.at[0, "promoter_dilution_pp"] - 11.41) < 0.05
    assert d.at[0, "ofs_pct"] == 100
