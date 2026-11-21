"""Offline tests for src/report.py: synthetic IPOs, no network, no Claude calls."""
import numpy as np
import pandas as pd
import pytest

from src import report as R


def synth(n=700, seed=1) -> pd.DataFrame:
    """Gain depends on QIB subscription and PAT margin plus noise, so a sound model should find it."""
    rng = np.random.default_rng(seed)
    qib = np.exp(rng.normal(2.6, 1.6, n))
    retail = qib ** 0.6 * np.exp(rng.normal(0, 0.4, n))
    total = qib ** 0.8 * np.exp(rng.normal(0, 0.3, n))
    pat = rng.normal(9, 7, n)
    gain = 12 * np.log1p(qib) - 25 + 0.5 * pat + rng.normal(0, 8, n)
    dates = pd.to_datetime("2022-01-03") + pd.to_timedelta(np.sort(rng.integers(0, 1800, n)), unit="D")
    d = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)], "company": [f"Co {i}" for i in range(n)],
        "sector": rng.choice(["Chemicals", "IT", "Auto", "Banks", "Retail"], n), "listing_date": dates,
        "listing_gain_pct": gain, "sub_qib_x": qib, "sub_retail_x": retail, "sub_total_x": total, "sub_nii_x": qib * 0.7,
        "pat_margin_fy0_pct": pat, "revenue_cagr_pct": rng.normal(25, 15, n), "ronw_fy0_pct": rng.normal(18, 8, n),
        "debt_to_equity_fy0": rng.uniform(0, 3, n), "pe_premium_to_peers_pct": rng.normal(0, 40, n),
        "ofs_pct": rng.uniform(0, 100, n), "issue_size_cr": np.exp(rng.normal(6.2, 0.9, n)), "loss_making_flag": (pat < 0).astype(float),
        "nifty_ret_1m_pct": rng.normal(1, 4, n), "avg_listing_gain_prev_5_pct": rng.normal(15, 10, n),
        "pat_cagr_pct": rng.normal(40, 30, n), "ebitda_margin_fy0_pct": pat * 1.8, "cfo_to_pat": rng.normal(0.9, 0.3, n),
        "pe_at_issue": rng.uniform(10, 60, n), "proceeds_debt_repayment_pct": rng.uniform(0, 80, n),
        "ipo_premium_to_preipo_pct": rng.normal(5, 20, n), "lot_size": rng.integers(20, 200, n), "issue_price": rng.uniform(80, 600, n),
    })
    d["target_gain"] = (d["listing_gain_pct"] > 0).astype(int)
    return R.prepare(d)


@pytest.fixture(scope="module")
def data():
    return synth()


@pytest.fixture(scope="module")
def models(data):
    return R.fit_models(data)


def a_row(data, **over) -> pd.Series:
    row = data[data["listing_date"] >= "2026-01-01"].iloc[0].copy()
    row["price_high"], row["lot_size"] = 200.0, 75
    for k, v in over.items():
        row[k] = v
    return row


# ---- stages ---------------------------------------------------------------- #
def test_stage_labels():
    assert R.stage_of({}, False) == "pre-bid"
    assert R.stage_of({"sub_qib_x": 0.0}, True) == "pre-bid"
    assert R.stage_of({"sub_qib_x": 3.0}, False) == "provisional"
    assert R.stage_of({"sub_qib_x": 3.0}, True) == "final"
    assert R.stage_of({"sub_total_x": 3.0}, True) == "final"


def test_probability_stages_and_monotone_in_qib(models, data):
    row = a_row(data)
    sub = {"sub_qib_x": 20.0, "sub_total_x": 15.0, "sub_retail_x": 5.0}
    pre = R.estimate(models, row, {}, False)
    prov = R.estimate(models, row, sub, False)
    fin = R.estimate(models, row, sub, True)
    assert (pre["stage"], prov["stage"], fin["stage"]) == ("pre-bid", "provisional", "final")
    assert (pre["headline"], prov["headline"], fin["headline"]) == ("pre-bid", "pre-bid", "final")
    assert prov["floor"] and not fin["floor"] and not pre["floor"]
    lo = R.estimate(models, row, {"sub_qib_x": 1.0, "sub_total_x": 1.0, "sub_retail_x": 1.0}, True)
    hi = R.estimate(models, row, {"sub_qib_x": 100.0, "sub_total_x": 60.0, "sub_retail_x": 20.0}, True)
    assert lo["prob"] < fin["prob"] < hi["prob"]
    assert 0 <= pre["prob"] <= 1 and pre["wide"]


def test_provisional_headline_is_the_prebid_estimate(models, data):
    """While bidding is open the headline must not move with live subscription; the demand model is 'if closed'."""
    row = a_row(data)
    pre = R.estimate(models, row, {}, False)
    for q in (0.5, 5.0, 80.0):
        prov = R.estimate(models, row, {"sub_qib_x": q, "sub_total_x": q * 0.7, "sub_retail_x": q * 0.4}, False)
        assert prov["prob"] == pytest.approx(pre["prob"])
        assert prov["gain"] == pytest.approx(pre["gain"])
        assert prov["prices"] == pytest.approx(pre["prices"])
        fin = R.estimate(models, row, {"sub_qib_x": q, "sub_total_x": q * 0.7, "sub_retail_x": q * 0.4}, True)
        assert prov["if_closed"]["prob"] == pytest.approx(fin["prob"])
        assert prov["if_closed"]["gain"] == pytest.approx(fin["gain"])
    assert pre["if_closed"] is None and R.estimate(models, row, {"sub_qib_x": 9.0}, True)["if_closed"] is None
    assert prov["qib_needed"] == pre["qib_needed"]                 # needed QIB depends on the issue, not on live figures


def test_allotment_and_profit_hidden_until_bidding_closed(models, data):
    row = a_row(data)
    sub = {"sub_qib_x": 20.0, "sub_total_x": 15.0, "sub_retail_x": 5.0}
    for closed, stage_sub in ((False, {}), (False, sub)):
        e = R.estimate(models, row, stage_sub, closed)
        assert np.isnan(e["p_allot"]) and np.isnan(e["profit_per_lot"])
    fin = R.estimate(models, row, sub, True)
    assert fin["p_allot"] == pytest.approx(0.2) and np.isfinite(fin["profit_per_lot"])


def test_provisional_report_leads_with_qib_needed_and_hides_profit(models, data):
    rep = R.build_report(a_row(data), None, None, models, data, {"sub_qib_x": 4.0, "sub_total_x": 3.0, "sub_retail_x": 2.0}, False)
    md = R.to_markdown(rep)
    sec = md[md.index("## 2."):md.index("## 3.")]
    assert sec.index("QIB subscription needed") < sec.index("Chance of listing above issue price")
    assert "If bidding closed at today's figures" in sec
    assert "once bidding has closed" in sec and "expected profit per lot: ₹" not in sec
    assert rep.verdict == "Wait for subscription"


def test_qib_needed_is_increasing(models, data):
    need = R.qib_needed(models, a_row(data))
    vals = [need[t] for t in (0.70, 0.85, 0.95)]
    assert all(v is None or v > 0 for v in vals)
    seen = [v for v in vals if v is not None]
    assert seen == sorted(seen)


# ---- gain ranges ----------------------------------------------------------- #
@pytest.mark.parametrize("gm", ["A", "B"])
def test_quantiles_are_ordered(models, data, gm):
    for q in (0.5, 3, 12, 40, 150):
        e = R.estimate(models, a_row(data), {"sub_qib_x": q, "sub_total_x": q * 0.7, "sub_retail_x": q * 0.4}, True, gain_model=gm)
        g = [e["gain"][k] for k in R.QKEYS]
        assert g == sorted(g)
        p = [e["prices"][k] for k in R.QKEYS]
        assert p == sorted(p)


def test_model_a_ordering_for_many_rows(models, data):
    _, test = R.split(data)
    pred = R._predict_a(models.quantile_a, test)
    stacked = np.column_stack([pred[k] for k in R.QKEYS])
    assert (np.diff(stacked, axis=1) >= -1e-9).all()


def test_interval_coverage_on_synthetic_data(models, data):
    _, test = R.split(data)
    pred = R._predict_a(models.quantile_a, test)
    s = R.gain_scores(test["listing_gain_pct"].to_numpy(), pred)
    assert 0.35 <= s["cov_25_75"] <= 0.70
    assert 0.65 <= s["cov_10_90"] <= 0.95
    assert s["spearman"] > 0.7                      # the signal is strong by construction


def test_expected_mean_uses_quantile_weights(models, data):
    e = R.estimate(models, a_row(data), {"sub_qib_x": 10.0, "sub_total_x": 8.0, "sub_retail_x": 4.0}, True)
    assert e["gain_mean"] == pytest.approx(float(np.dot(R.QWEIGHTS, [e["gain"][k] for k in R.QKEYS])))
    assert R.QWEIGHTS.sum() == pytest.approx(1.0)


# ---- verdict --------------------------------------------------------------- #
@pytest.mark.parametrize("prob,qib,stage,want", [
    (0.90, 12, "final", "Apply"), (0.85, 10, "final", "Apply"), (0.86, 9.9, "final", "Apply with caution"),
    (0.84, 30, "final", "Apply with caution"), (0.54, 30, "final", "Skip"), (0.90, 2.9, "final", "Skip"),
    (0.70, 5, "final", "Apply with caution"), (0.40, 1, "provisional", "Wait for subscription"),
    (0.90, 15, "provisional", "Apply"), (0.7, None, "pre-bid", "Wait for subscription"),
])
def test_verdict_thresholds(prob, qib, stage, want):
    label, reasons = R.verdict(prob, qib, stage)
    assert label == want
    assert reasons and all(isinstance(x, str) for x in reasons)


# ---- allotment maths ------------------------------------------------------- #
def test_allotment_probability():
    assert R.allotment_prob(0.4) == 1.0
    assert R.allotment_prob(1.0) == 1.0
    assert R.allotment_prob(4.0) == pytest.approx(0.25)
    assert R.allotment_prob(50) == pytest.approx(0.02)
    assert np.isnan(R.allotment_prob(None)) and np.isnan(R.allotment_prob(0))


def test_profit_per_lot_matches_formula(models, data):
    e = R.estimate(models, a_row(data), {"sub_qib_x": 20.0, "sub_total_x": 12.0, "sub_retail_x": 5.0}, True)
    assert e["p_allot"] == pytest.approx(0.2)
    assert e["lot_value"] == pytest.approx(75 * 200.0)
    assert e["profit_per_lot"] == pytest.approx(0.2 * e["gain_mean"] / 100 * 75 * 200.0)


# ---- backtest, tables and markdown ---------------------------------------- #
def test_backtest_outputs(models, data):
    bt = R.backtest(models, data)
    for k in ("demand", "prebid", "gain_a", "gain_b", "gain_prebid_knn", "strategies", "calibration_demand", "baselines"):
        assert k in bt
    assert bt["demand"]["auc"] > 0.8
    assert list(bt["strategies"].index)[:2] == ["Always apply", "Total subscription > 10x"]
    assert {"QIB > 3x", "QIB > 5x", "QIB > 10x", "QIB > 20x", "QIB > 50x"} <= set(bt["strategies"].index)
    assert "expected_gain_per_application_pct" in bt["strategies"].columns
    assert not any("allotment_weighted" in c for c in bt["strategies"].columns)
    cal = bt["calibration_demand"]
    assert cal["n"].sum() == bt["demand"]["n"] and cal["predicted"].is_monotonic_increasing


def test_strategy_hit_rate_rises_with_threshold(data):
    _, test = R.split(data)
    t = R.strategy_table(test)
    assert t.loc["QIB > 50x", "hit_rate"] >= t.loc["Always apply", "hit_rate"]
    assert t.loc["Always apply", "n_applied"] == len(test)


def test_scorecard_scores_in_range(data):
    sc = R.pillar_scores(a_row(data), data)
    assert list(sc.index) == list(R.PILLARS)
    assert sc["score_0_10"].dropna().between(0, 10).all()
    assert sc.loc["Demand", "train_auc"] > 0.7


def test_similar_returns_ten_with_gains(models, data):
    s = R.similar(models, a_row(data), 10)
    assert len(s) == 10 and s["listing_gain_pct"].notna().all()
    assert s["distance"].is_monotonic_increasing


def test_report_and_markdown(models, data):
    bt = R.backtest(models, data)
    rep = R.build_report(a_row(data), None, None, models, data, {"sub_qib_x": 25.0, "sub_total_x": 18.0, "sub_retail_x": 6.0, "sub_nii_x": 20.0}, True, backtest_res=bt)
    assert rep.verdict in {"Apply", "Apply with caution", "Skip", "Wait for subscription"}
    assert not rep.band_table.empty and len(rep.similar) == 10
    md = R.to_markdown(rep)
    heads = [f"## {i}." for i in range(1, 9)]
    pos = [md.index(h) for h in heads]
    assert pos == sorted(pos)                         # sections appear in the required order
    assert "Chance of listing above issue price" in md and "QIB subscription needed" in md


def test_prepare_treats_zero_subscription_as_missing():
    d = R.prepare(pd.DataFrame({"sub_qib_x": [0.0, 5.0], "sub_total_x": [0.0, 3.0], "sub_retail_x": [1.0, 0.0], "sector": [None, "IT"],
                                "listing_date": ["2024-01-01", "2024-02-01"], "issue_size_cr": [100, 200]}))
    assert d["sub_qib_x"].isna().tolist() == [True, False] and d["sub_retail_x"].isna().tolist() == [False, True]
    assert d["sector"].tolist() == ["unknown", "IT"]


# ---- no leakage ----------------------------------------------------------- #
def test_scrambling_2026_outcomes_changes_nothing(data):
    base = R.fit_models(data)
    scrambled = data.copy()
    test_mask = scrambled["listing_date"] >= "2026-01-01"
    assert test_mask.sum() > 50
    rng = np.random.default_rng(99)
    scrambled.loc[test_mask, "listing_gain_pct"] = rng.permutation(scrambled.loc[test_mask, "listing_gain_pct"].to_numpy()) + 500
    again = R.fit_models(scrambled)
    row = a_row(data)
    sub = {"sub_qib_x": 15.0, "sub_total_x": 9.0, "sub_retail_x": 4.0}
    np.testing.assert_allclose(base.demand[-1].coef_, again.demand[-1].coef_)
    np.testing.assert_allclose(base.prebid[-1].coef_, again.prebid[-1].coef_)
    e1, e2 = R.estimate(base, row, sub, True), R.estimate(again, row, sub, True)
    assert e1["prob"] == pytest.approx(e2["prob"])
    assert e1["gain"] == pytest.approx(e2["gain"])
    assert R.estimate(base, row, {}, False)["gain"] == pytest.approx(R.estimate(again, row, {}, False)["gain"])
    assert base.kind_a == again.kind_a and base.default_gain_model == again.default_gain_model
    assert base.widen == again.widen                  # range widening uses 2024-2025 walk-forward residuals only
    np.testing.assert_allclose(base.aux_total.coef_, again.aux_total.coef_)
    assert len(base.train) == len(again.train)
    assert base.train["listing_date"].max() < pd.Timestamp("2026-01-01")


# ---- walk-forward, calibrated ranges, production refit ------------------- #
@pytest.fixture(scope="module")
def wf_raw(data):
    return R.walk_forward_raw(data)


def test_walk_forward_trains_only_on_earlier_years(data, wf_raw):
    assert sorted(wf_raw) == [2024, 2025, 2026]
    for y, r in wf_raw.items():
        assert (r["test"]["listing_date"].dt.year == y).all()
        earlier = int((data["listing_date"] < f"{y}-01-01").sum())
        assert r["n_train"] == earlier


def test_walk_forward_tables_have_every_year_and_pooled(data, wf_raw):
    wf = R.walk_forward(data, wf_raw)
    assert list(wf["prob"].index) == ["2024", "2025", "2026", "pooled"]
    assert wf["prob"].loc["pooled", "n"] == sum(wf["prob"].loc[y, "n"] for y in ("2024", "2025", "2026"))
    g = wf["gain"]
    assert {m for m, _ in g.index} == {"A", "B", "prebid", "prebid_knn"}
    assert ("A", "pooled") in g.index and wf["prob"].loc["pooled", "demand_auc"] > 0.8


def test_widening_only_widens_and_hits_targets(data, wf_raw):
    widen = R.calibrate(list(wf_raw.values()))
    for model, bands in widen.items():
        for band, w in bands.items():
            assert w["inner"] >= 0 and w["outer"] >= 0
    assert set(widen["A"]) == set(R.BANDS) and set(widen["prebid"]) == {"pre-bid"}
    # coverage after widening on the same residuals reaches the targets (within sampling noise)
    raws = list(wf_raw.values())
    y = np.concatenate([r["gain_d"] for r in raws])
    bands = np.concatenate([r["band_d"] for r in raws])
    raw = {k: np.concatenate([r["A"][k] for r in raws]) for k in R.QKEYS}
    cal = R._apply_arrays(raw, bands, "A", widen)
    s_raw, s_cal = R.gain_scores(y, raw), R.gain_scores(y, cal)
    assert s_cal["cov_25_75"] >= s_raw["cov_25_75"] - 1e-9 and s_cal["cov_10_90"] >= s_raw["cov_10_90"] - 1e-9
    assert abs(s_cal["cov_25_75"] - 0.50) < 0.08 and abs(s_cal["cov_10_90"] - 0.80) < 0.08


def test_apply_widen_keeps_order_and_leaves_median():
    q = {"p10": -5.0, "p25": 0.0, "p50": 4.0, "p75": 9.0, "p90": 15.0}
    widen = {"A": {"10-100x": {"inner": 3.0, "outer": 6.0, "n": 50}}}
    w = R.apply_widen(q, "A", "10-100x", widen)
    assert w["p50"] == 4.0 and w["p25"] == -3.0 and w["p75"] == 12.0 and w["p10"] == -11.0 and w["p90"] == 21.0
    assert list(w.values()) == sorted(w.values())
    assert R.apply_widen(q, "A", ">100x", widen) == q                    # no stored shift for that band


def test_qib_band_edges():
    assert [R.qib_band(v) for v in (None, 0, 3, 9.99, 10, 100, 100.5)] == ["pre-bid", "pre-bid", "<10x", "<10x", "10-100x", "10-100x", ">100x"]


def test_prebid_ranges_are_widened(models, data):
    e = R.estimate(models, a_row(data), {}, False)
    pts = R.prebid_points(models, a_row(data).to_frame().T)
    raw = R.mix_quantiles(pts["qk"])
    raw_width = raw["p90"][0] - raw["p10"][0]
    assert e["gain"]["p90"] - e["gain"]["p10"] >= raw_width - 1e-9
    w = models.widen["prebid"]["pre-bid"]
    assert w["inner"] >= 0 and w["outer"] >= 0 and set(models.widen) >= {"A", "B", "prebid", "prebid_knn"}


def test_production_refit_uses_every_row_but_published_does_not(data, wf_raw):
    pub = R.fit_models(data, wf_raw=wf_raw)
    prod = R.fit_models(data, cutoff=None, wf_raw=wf_raw)
    assert pub.cutoff == "2026-01-01" and prod.cutoff is None
    assert pub.train["listing_date"].max() < pd.Timestamp("2026-01-01")
    assert len(prod.train) == int(data["listing_gain_pct"].notna().sum()) > len(pub.train)
    assert (prod.train["listing_date"] >= "2026-01-01").any()
    # the published backtest is scored on 2026 only
    assert R.backtest(pub, data)["n_test"] == int((data["listing_date"] >= "2026-01-01").sum())


def test_split_without_cutoff_puts_everything_in_train(data):
    tr, te = R.split(data, None)
    assert len(te) == 0 and len(tr) == int(data["listing_gain_pct"].notna().sum())
    tr, te = R.split(data, "2025-01-01", "2026-01-01")
    assert (te["listing_date"].dt.year == 2025).all() and (tr["listing_date"] < "2025-01-01").all()


# ---- pre-bid estimate from a QIB forecast --------------------------------- #
class FixedForecast(R.QibForecaster):
    """A pluggable forecaster: always predicts the same final QIB subscription."""

    def __init__(self, qib, sd=1.0):
        self.mu, self.sd = float(np.log1p(qib)), sd
        self.seen = None

    def fit(self, train, features):
        return self

    def forecast(self, rows, snapshots=None):
        self.seen = snapshots
        return np.full(len(rows), self.mu), np.full(len(rows), self.sd)


def test_qib_forecaster_is_fitted_on_training_rows_only(data):
    base = R.fit_models(data)
    scrambled = data.copy()
    mask = scrambled["listing_date"] >= "2026-01-01"
    scrambled.loc[mask, "sub_qib_x"] = scrambled.loc[mask, "sub_qib_x"].to_numpy()[::-1] * 7 + 50
    scrambled["l_qib"] = np.log1p(scrambled["sub_qib_x"])
    again = R.fit_models(scrambled)
    np.testing.assert_allclose(base.forecaster.pipe[-1].coef_, again.forecaster.pipe[-1].coef_)
    assert base.forecaster.sd_ == pytest.approx(again.forecaster.sd_)
    assert base.forecaster.n_ == int(base.train["l_qib"].notna().sum()) and base.forecaster.features == R.PREBID_NUM
    assert R.fit_models(data, cutoff=None).forecaster.n_ > base.forecaster.n_              # production uses every row


def test_forecast_distribution_points_and_weights():
    assert len(R.POINT_Z) == 21 == R.N_POINTS and R.POINT_Z[0] == -2.5 and R.POINT_Z[-1] == 2.5
    assert R.POINT_W.sum() == pytest.approx(1.0) and R.POINT_W[10] == R.POINT_W.max()
    assert R.POINT_W == pytest.approx(R.POINT_W[::-1])                                     # symmetric around the forecast mean


def test_prebid_mixture_uses_the_forecast(models, data):
    row = a_row(data)
    probs, gains = [], []
    for q in (1.0, 10.0, 200.0):
        models.forecaster = FixedForecast(q)
        e = R.estimate(models, row, {}, False)
        probs.append(e["prob"]); gains.append(e["gain"]["p50"])
        assert e["gain_model"] == "QIB forecast" and e["wide"] and e["stage"] == "pre-bid"
    models.forecaster = R.RidgeQibForecaster().fit(models.train, R.PREBID_NUM)
    assert probs[0] < probs[1] < probs[2] and gains[0] < gains[1] < gains[2]               # a bigger forecast means a better outlook


def test_mixture_probability_and_quantiles_are_consistent(models, data):
    pts = R.prebid_points(models, a_row(data).to_frame().T)
    pk, qk = pts["pk"], pts["qk"]
    p = R.mix_probability(pk)[0]
    assert pk.min() <= p <= pk.max() and p == pytest.approx(float((pk[0] * R.POINT_W).sum()))
    q = R.mix_quantiles(qk)
    vals = [q[k][0] for k in R.QKEYS]
    assert vals == sorted(vals)
    # twenty-one identical points must reproduce that point's own quantiles
    same = np.repeat(qk[:, 10:11, :], R.N_POINTS, axis=1)
    q2 = R.mix_quantiles(same)
    np.testing.assert_allclose([q2[k][0] for k in R.QKEYS], same[0, 0], atol=0.2)
    # the mixture is wider than its middle point because the forecast is uncertain
    assert vals[-1] - vals[0] >= qk[0, 10, -1] - qk[0, 10, 0] - 1e-6


def test_prebid_keeps_knn_as_a_table_only(models, data):
    e = R.estimate(models, a_row(data), {}, False)
    assert len(e["similar"]) == 10 and e["headline"] == "pre-bid" and e["gain_model"] != "kNN"
    rep = R.build_report(a_row(data), None, None, models, data, {}, False)
    assert not any("most similar past IPOs listed above issue" in x for x in rep.reasons)


def test_forecaster_interface_and_snapshot_stub(data):
    with pytest.raises(NotImplementedError):
        R.QibForecaster().forecast(data)
    from src.qibforecast import SnapshotQibForecaster
    train, _ = R.split(data)
    stub = SnapshotQibForecaster().fit(train, R.PREBID_NUM)
    plain = R.RidgeQibForecaster().fit(train, R.PREBID_NUM)
    rows = data.head(5)
    np.testing.assert_allclose(stub.forecast(rows, {"day": 1, "sub_qib_x": 4.0})[0], plain.forecast(rows)[0])    # snapshots ignored for now
    assert "TODO" in SnapshotQibForecaster.__doc__ and "snapshots.csv" in SnapshotQibForecaster.__doc__


def test_a_custom_forecaster_receives_snapshots(models, data):
    models.forecaster = FixedForecast(30.0)
    R.estimate(models, a_row(data), {}, False, snapshots={"day": 1, "sub_qib_x": 3.0})
    assert models.forecaster.seen == {"day": 1, "sub_qib_x": 3.0}
    models.forecaster = R.RidgeQibForecaster().fit(models.train, R.PREBID_NUM)


# ---- low pre-bid probabilities are shown as the base rate ------------------ #
def test_prebid_probability_text_rule():
    assert R.prebid_prob_text(0.55, 0.74) == ("about the base rate (74%)", False)
    assert R.prebid_prob_text(0.799, 0.74)[1] is False
    assert R.prebid_prob_text(0.80, 0.74) == ("80%", True)
    assert R.prebid_prob_text(0.93, 0.74) == ("93%", True)
    assert R.prebid_prob_text(float("nan"), 0.74) == ("–", False)


def test_base_rate_comes_from_training_rows_and_drives_the_text(models, data):
    train, _ = R.split(data)
    assert models.base_rate == pytest.approx(float((train["listing_gain_pct"] > 0).mean()))
    assert R.fit_models(data, cutoff=None).base_rate == pytest.approx(float((data["listing_gain_pct"] > 0).mean()))
    models.forecaster = FixedForecast(1.0)                                              # a weak forecast: low pre-bid probability
    e = R.estimate(models, a_row(data), {}, False)
    assert e["prob"] < 0.80 and not e["prob_precise"] and e["prob_text"] == f"about the base rate ({models.base_rate:.0%})"
    md = R.to_markdown(R.build_report(a_row(data), None, None, models, data, {}, False))
    assert "about the base rate" in md and f"model value {e['prob']:.0%}" in md
    models.forecaster = FixedForecast(400.0)                                            # a strong forecast: precise figure
    e = R.estimate(models, a_row(data), {}, False)
    assert e["prob"] >= 0.80 and e["prob_precise"] and e["prob_text"] == f"{e['prob']:.0%}"
    models.forecaster = R.RidgeQibForecaster().fit(models.train, R.PREBID_NUM)
    fin = R.estimate(models, a_row(data), {"sub_qib_x": 2.0, "sub_total_x": 1.5, "sub_retail_x": 1.0}, True)
    assert fin["prob_precise"] and fin["prob_text"] == f"{fin['prob']:.0%}"              # final figures are always precise


# ---- walk-forward: new method vs old, and isotonic calibration ------------- #
def test_walk_forward_compares_new_and_old_prebid(data, wf_raw):
    wf = R.walk_forward(data, wf_raw)
    cols = set(wf["prob"].columns)
    assert {"prebid_auc", "prebid_brier", "prebid_acc", "old_prebid_auc", "old_prebid_brier", "old_prebid_acc"} <= cols
    assert {"demand_brier_raw", "demand_brier_cal", "demand_ece_raw", "demand_ece_cal"} <= cols
    assert {("prebid", "pooled"), ("prebid_knn", "pooled")} <= set(wf["gain"].index)
    assert list(wf["forecast"].index) == ["2024", "2025", "2026", "pooled"]
    rule = wf["prebid_rule"]
    assert rule["n_high"] + rule["n_low"] == int(wf["prob"].loc["pooled", "n"])
    assert len(wf["calibration_before"]) == len(wf["calibration_after"]) > 0


def test_isotonic_is_fitted_leave_one_year_out(wf_raw):
    maps = R.loyo_isotonic(wf_raw)
    assert set(maps) == set(wf_raw)
    grid = np.linspace(0.05, 0.95, 19)
    before = {y: maps[y].predict(grid) for y in maps}
    changed = {y: {**r, "yd": 1 - r["yd"]} if y == 2026 else r for y, r in wf_raw.items()}      # flip only 2026's outcomes
    after = R.loyo_isotonic(changed)
    np.testing.assert_allclose(after[2026].predict(grid), before[2026])                  # 2026's own map ignores its own labels
    assert not np.allclose(after[2024].predict(grid), before[2024])                      # other years' maps use them
    for m in maps.values():
        out = m.predict(grid)
        assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1      # monotone, valid probabilities


def test_isotonic_gate_and_modes(models, wf_raw, monkeypatch):
    raws = [wf_raw[y] for y in sorted(wf_raw)]
    g = R.iso_gate(raws)
    assert {"brier_raw", "brier_cal", "ece_raw", "ece_cal", "applied", "reason"} <= set(g)
    assert g["applied"] == (g["brier_raw"] - g["brier_cal"] >= R.ISO_MIN_BRIER_GAIN)
    assert R.iso_gate(raws[:1])["applied"] is False                                       # one year gives no evidence
    monkeypatch.setattr(R, "ISOTONIC", "on")
    assert R.iso_gate(raws)["applied"] is True and "forced" in R.iso_gate(raws)["reason"]
    forced = R.fit_models(synth(), wf_raw=wf_raw)
    assert forced.iso_demand is not None and forced.iso_demand is forced.iso_candidate
    monkeypatch.setattr(R, "ISOTONIC", "off")
    off = R.fit_models(synth(), wf_raw=wf_raw)
    assert off.iso_demand is None and off.iso_candidate is not None                       # fitted and kept, just not applied
    p = np.array([0.2, 0.6, 0.9])
    np.testing.assert_allclose(off.cal_prob(p), p)                                       # identity when not applied
    assert (np.diff(forced.cal_prob(np.linspace(0.05, 0.95, 10))) >= -1e-12).all()


def test_forced_isotonic_changes_demand_probability_and_qib_needed(wf_raw, monkeypatch):
    d = synth()
    monkeypatch.setattr(R, "ISOTONIC", "off")
    off = R.fit_models(d, wf_raw=wf_raw)
    monkeypatch.setattr(R, "ISOTONIC", "on")
    on = R.fit_models(d, wf_raw=wf_raw)
    row = a_row(d)
    sub = {"sub_qib_x": 20.0, "sub_total_x": 15.0, "sub_retail_x": 5.0}
    e_off, e_on = R.estimate(off, row, sub, True), R.estimate(on, row, sub, True)
    assert e_on["prob"] == pytest.approx(float(on.cal_prob(e_off["prob"])))
    assert e_on["qib_needed"] != e_off["qib_needed"] or e_on["prob"] != e_off["prob"]


def test_backtest_reports_isotonic_before_and_after(models, data):
    bt = R.backtest(models, data)
    assert {"demand_raw", "demand_isotonic", "prebid_old", "gain_prebid", "gain_prebid_knn", "ece", "iso_gate", "qib_forecast"} <= set(bt)
    assert set(bt["ece"]) == {"demand_raw", "demand_isotonic"}
    assert len(bt["calibration_demand_raw"]) == len(bt["calibration_demand"])


def test_new_prebid_widening_and_walk_forward_are_leak_free(data):
    base = R.fit_models(data)
    scrambled = data.copy()
    m = scrambled["listing_date"] >= "2026-01-01"
    scrambled.loc[m, "listing_gain_pct"] = np.random.default_rng(3).permutation(scrambled.loc[m, "listing_gain_pct"].to_numpy()) + 300
    again = R.fit_models(scrambled)
    assert base.widen == again.widen and base.iso_gate == again.iso_gate and base.base_rate == again.base_rate
    row, sub = a_row(data), {}
    e1, e2 = R.estimate(base, row, sub, False), R.estimate(again, row, sub, False)
    assert e1["prob"] == pytest.approx(e2["prob"]) and e1["gain"] == pytest.approx(e2["gain"])


# ---- stale text ------------------------------------------------------------ #
def test_report_header_and_app_text_say_live_models_are_refit():
    import pathlib
    md = R.to_markdown(R.build_report(a_row(synth()), None, None, R.fit_models(synth()), synth(), {}, False))
    assert "refit on all IPOs to date" in md.split("\n")[2] and "walk-forward" in md.split("\n")[2]
    app = pathlib.Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "Live estimates use models refit on all IPOs to date" in app and "walk-forward" in app
    assert "Models learn from IPOs listed 2022-2025 and are scored on 2026" not in app


def test_readme_documents_windows_task_scheduler():
    import pathlib
    readme = pathlib.Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert "*/30 9-17 * * 1-5" in readme                                                   # the cron line stays
    assert "Task Scheduler" in readme and "Register-ScheduledTask" in readme and "src.build snapshot" in readme
    assert "-RepetitionInterval" in readme and "Monday" in readme
