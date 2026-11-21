"""Listing-gain probability, expected gain, backtest and report assembly.

All numbers come from small models fitted on IPOs listed 2022-2025 only (never on 2026 rows).
Claude only extracts text and printed figures elsewhere; every ratio, probability and gain here is
computed in code.

    models = fit_models(prepare(history))
    est    = estimate(models, row, sub={"sub_qib_x": 12.0, ...}, closed=True)
    bt     = backtest(models, prepare(history))
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import TEST_START

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QKEYS = ("p10", "p25", "p50", "p75", "p90")
# share of the distribution each quantile stands for; used to turn the five quantiles into a mean
QWEIGHTS = np.array([0.175, 0.20, 0.25, 0.20, 0.175])

DEMAND_NUM = ["l_qib", "l_total", "l_retail", "pat_margin_fy0_pct"]
PREBID_NUM = ["pat_margin_fy0_pct", "revenue_cagr_pct", "ronw_fy0_pct", "debt_to_equity_fy0", "pe_premium_to_peers_pct",
              "ofs_pct", "log_issue_size", "loss_making_flag", "nifty_ret_1m_pct", "avg_listing_gain_prev_5_pct"]
SUB_COLS = ("sub_qib_x", "sub_nii_x", "sub_retail_x", "sub_total_x")
APPLICATION_RS = 15_000          # notional retail application when a lot value is unknown
MIN_NARROW_ROWS = 15
PAT_SPLIT = 8.0


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add model columns. A subscription of exactly 0 means 'not recorded', so it becomes missing."""
    d = df.copy()
    d["listing_date"] = pd.to_datetime(d["listing_date"], errors="coerce")
    for c in SUB_COLS:
        col = pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
        d[c] = col.where(col > 0)
    d["l_qib"], d["l_total"], d["l_retail"] = (np.log1p(d["sub_qib_x"]), np.log1p(d["sub_total_x"]), np.log1p(d["sub_retail_x"]))
    d["log_issue_size"] = np.log1p(pd.to_numeric(d.get("issue_size_cr"), errors="coerce"))
    d["sector"] = d["sector"].fillna("unknown") if "sector" in d else "unknown"
    return d


def split(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = listed before 2026-01-01, test = listed 2026 onward. Only rows with an outcome."""
    d = d[d["listing_gain_pct"].notna() & d["listing_date"].notna()]
    cut = pd.Timestamp(TEST_START)
    return d[d["listing_date"] < cut], d[d["listing_date"] >= cut]


def allotment_prob(retail_x) -> float:
    """Chance a retail application gets one lot: about min(1, 1 / retail subscription)."""
    r = pd.to_numeric(pd.Series([retail_x]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(r) or r <= 0 else float(min(1.0, 1.0 / r))


class Winsor(BaseEstimator, TransformerMixin):
    """Clip each column to its training 1st-99th percentile so one odd RHP figure cannot dominate."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _num_pipe(model):
    return make_pipeline(Winsor(), SimpleImputer(strategy="median"), StandardScaler(), model)


def _with_sector(num_cols: list[str], model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", make_pipeline(Winsor(), SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("sec", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"), ["sector"]),
    ])
    return Pipeline([("pre", pre), ("m", model)])


def _frame(row: pd.Series | pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = row.to_frame().T if isinstance(row, pd.Series) else row
    X = X.copy()
    for c in cols:
        if c not in X:
            X[c] = np.nan
    return X


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class Models:
    demand: Pipeline                      # P(gain) from subscription + PAT margin
    prebid: Pipeline                      # P(gain) from fundamentals and market mood
    quantile_a: dict[str, Pipeline]       # Model A: conditional quantiles of the gain
    kind_a: str                           # "linear" or "gbm"
    train: pd.DataFrame
    aux_total: LinearRegression           # l_total ~ l_qib
    aux_retail: LinearRegression          # l_retail ~ l_qib
    zmean: pd.Series
    zstd: pd.Series
    default_gain_model: str = "A"         # "A" or "B", chosen on 2025 validation inside the training set
    info: dict = field(default_factory=dict)


def _fit_quantiles(train: pd.DataFrame, kind: str) -> dict[str, Pipeline]:
    out = {}
    cols = DEMAND_NUM
    for q, k in zip(QUANTILES, QKEYS):
        m = QuantileRegressor(quantile=q, alpha=0.01, solver="highs") if kind == "linear" else \
            GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=80, max_depth=2, learning_rate=0.05,
                                      min_samples_leaf=15, subsample=0.8, random_state=0)
        pipe = _with_sector(cols, m)
        pipe.fit(train[cols + ["sector"]], train["listing_gain_pct"])
        out[k] = pipe
    return out


def _pinball(y: np.ndarray, pred: dict[str, np.ndarray]) -> float:
    tot = 0.0
    for q, k in zip(QUANTILES, QKEYS):
        e = y - pred[k]
        tot += float(np.mean(np.maximum(q * e, (q - 1) * e)))
    return tot / len(QUANTILES)


def _predict_a(qm: dict[str, Pipeline], X: pd.DataFrame) -> dict[str, np.ndarray]:
    P = np.column_stack([qm[k].predict(X[DEMAND_NUM + ["sector"]]) for k in QKEYS])
    P = np.sort(P, axis=1)                    # quantile ordering p10 <= p25 <= ... <= p90
    return {k: P[:, i] for i, k in enumerate(QKEYS)}


def _empirical(train: pd.DataFrame, qib: float, pat_margin: float | None) -> tuple[dict[str, float], dict]:
    """Model B: gains of past IPOs with QIB subscription within x/2 to 2x, narrowed by PAT margin >= 8%."""
    pool = train[train["sub_qib_x"].between(qib / 2, qib * 2)]
    meta = {"window_n": len(pool), "narrowed": False, "widened": False}
    if len(pool) < MIN_NARROW_ROWS:                       # too few in the window: take the 30 nearest in log QIB
        pool = train.assign(_d=(train["l_qib"] - np.log1p(qib)).abs()).nsmallest(30, "_d")
        meta["widened"] = True
    if pat_margin is not None and not np.isnan(pat_margin):
        side = pool[pool["pat_margin_fy0_pct"] >= PAT_SPLIT] if pat_margin >= PAT_SPLIT else pool[pool["pat_margin_fy0_pct"] < PAT_SPLIT]
        if len(side) >= MIN_NARROW_ROWS:
            pool, meta["narrowed"] = side, True
    meta["n"] = len(pool)
    g = pool["listing_gain_pct"]
    return {k: float(g.quantile(q)) for q, k in zip(QUANTILES, QKEYS)}, meta


def fit_models(d: pd.DataFrame) -> Models:
    """Fit everything on rows listed before 2026. Rows from 2026 are never read for fitting."""
    train, _ = split(d)
    tr = train[train["sub_total_x"].notna() | train["sub_qib_x"].notna()]
    y = (tr["listing_gain_pct"] > 0).astype(int)
    demand = _num_pipe(LogisticRegression(C=0.5, max_iter=2000)).fit(tr[DEMAND_NUM], y)
    pb = train
    prebid = _num_pipe(LogisticRegression(C=0.05, max_iter=2000)).fit(pb[PREBID_NUM], (pb["listing_gain_pct"] > 0).astype(int))

    # choose Model A's flavour on a 2025 validation fold carved out of TRAINING rows
    fit_part, val = tr[tr["listing_date"] < "2025-01-01"], tr[tr["listing_date"] >= "2025-01-01"]
    best_kind, best_loss, losses = "linear", np.inf, {}
    if len(fit_part) > 80 and len(val) > 20:
        for kind in ("linear", "gbm"):
            losses[kind] = _pinball(val["listing_gain_pct"].to_numpy(), _predict_a(_fit_quantiles(fit_part, kind), val))
            if losses[kind] < best_loss:
                best_kind, best_loss = kind, losses[kind]
    qa = _fit_quantiles(tr, best_kind)

    # default gain model: A vs B on the same 2025 validation fold
    default = "A"
    if len(fit_part) > 80 and len(val) > 20:
        qa_val = _predict_a(_fit_quantiles(fit_part, best_kind), val)
        rows = [_empirical(fit_part, r.sub_qib_x, r.pat_margin_fy0_pct)[0] if pd.notna(r.sub_qib_x) else {k: np.nan for k in QKEYS}
                for r in val.itertuples()]
        qb = {k: np.array([r[k] for r in rows]) for k in QKEYS}
        ok = ~np.isnan(qb["p50"])
        if ok.sum() > 10:
            lb = _pinball(val["listing_gain_pct"].to_numpy()[ok], {k: v[ok] for k, v in qb.items()})
            la = _pinball(val["listing_gain_pct"].to_numpy()[ok], {k: v[ok] for k, v in qa_val.items()})
            default = "A" if la <= lb else "B"
            losses.update({"A_val": la, "B_val": lb})

    both = tr.dropna(subset=["l_qib", "l_total", "l_retail"])
    aux_t = LinearRegression().fit(both[["l_qib"]], both["l_total"])
    aux_r = LinearRegression().fit(both[["l_qib"]], both["l_retail"])
    z = train[PREBID_NUM + ["l_qib", "l_total"]]
    return Models(demand, prebid, qa, best_kind, train, aux_t, aux_r, z.mean(), z.std().replace(0, 1), default,
                  {"pinball_val": losses, "n_train": len(train), "n_demand_rows": len(tr)})


# --------------------------------------------------------------------------- #
# Similar IPOs
# --------------------------------------------------------------------------- #
def similar(models: Models, row: pd.Series, k: int = 10, use_demand: bool = False) -> pd.DataFrame:
    """The k closest past IPOs (z-scored distance on fundamentals; same sector counts as closer)."""
    cols = PREBID_NUM + (["l_qib", "l_total"] if use_demand else [])
    t = models.train
    z = lambda frame: ((frame[cols] - models.zmean[cols]) / models.zstd[cols]).clip(-4, 4)
    zt, zr = z(t), z(_frame(row, cols).apply(pd.to_numeric, errors="coerce")).iloc[0]
    diff = zt.sub(zr, axis=1)
    n_ok = diff.notna().sum(axis=1)
    dist = np.sqrt((diff.pow(2).sum(axis=1, skipna=True)) / n_ok.replace(0, np.nan))
    dist = dist.where(n_ok >= 3)
    if "sector" in row and row.get("sector") not in (None, "unknown") and not pd.isna(row.get("sector")):
        dist = dist.where(t["sector"] != row["sector"], dist * 0.8)
    out = t.assign(distance=dist).dropna(subset=["distance"]).nsmallest(k, "distance")
    return out[["symbol", "company", "sector", "listing_date", "sub_qib_x", "sub_total_x", "pat_margin_fy0_pct",
                "listing_gain_pct", "distance"]]


# --------------------------------------------------------------------------- #
# Estimate for one IPO
# --------------------------------------------------------------------------- #
STAGES = ("pre-bid", "provisional", "final")


def stage_of(sub: dict | None, closed: bool) -> str:
    has = bool(sub) and any(pd.notna(sub.get(c)) and sub.get(c) > 0 for c in ("sub_qib_x", "sub_total_x"))
    if not has:
        return "pre-bid"
    return "final" if closed else "provisional"


def _demand_row(row: pd.Series, sub: dict) -> pd.DataFrame:
    r = row.copy()
    for c in SUB_COLS:
        if sub.get(c) is not None and pd.notna(sub.get(c)) and sub.get(c) > 0:
            r[c] = float(sub[c])
    r["l_qib"], r["l_total"], r["l_retail"] = (np.log1p(r.get("sub_qib_x", np.nan)), np.log1p(r.get("sub_total_x", np.nan)),
                                               np.log1p(r.get("sub_retail_x", np.nan)))
    X = _frame(r, DEMAND_NUM + ["sector"]).copy()
    X[DEMAND_NUM] = X[DEMAND_NUM].apply(pd.to_numeric, errors="coerce")
    return X


def qib_needed(models: Models, row: pd.Series, targets=(0.70, 0.85, 0.95)) -> dict[float, float | None]:
    """Smallest QIB subscription for which the demand model reaches each probability, with total and
    retail subscription following QIB along their usual relationship."""
    grid = np.exp(np.linspace(np.log(0.5), np.log(3000), 240))
    lq = np.log1p(grid)
    q_df = pd.DataFrame({"l_qib": lq})
    X = pd.DataFrame({"l_qib": lq, "l_total": models.aux_total.predict(q_df),
                      "l_retail": models.aux_retail.predict(q_df),
                      "pat_margin_fy0_pct": pd.to_numeric(pd.Series([row.get("pat_margin_fy0_pct")]), errors="coerce").iloc[0]})
    p = models.demand.predict_proba(X[DEMAND_NUM])[:, 1]
    out = {}
    for t in targets:
        hit = np.where(p >= t)[0]
        out[t] = float(grid[hit[0]]) if len(hit) else None
    return out


def estimate(models: Models, row: pd.Series, sub: dict | None = None, closed: bool = False, gain_model: str | None = None) -> dict:
    """Probability, expected gain, price and allotment economics for one IPO."""
    sub = {k: v for k, v in (sub or {}).items() if v is not None and pd.notna(v)}
    stage = stage_of(sub, closed)
    gm = gain_model or models.default_gain_model
    price = pd.to_numeric(pd.Series([row.get("price_high", row.get("issue_price"))]), errors="coerce").iloc[0]
    lot = pd.to_numeric(pd.Series([row.get("lot_size")]), errors="coerce").iloc[0]
    res: dict = {"stage": stage, "gain_model": gm if stage != "pre-bid" else "kNN", "price": price}
    knn = similar(models, row, 10, use_demand=stage != "pre-bid")
    res["similar"] = knn
    res["knn_hit_rate"] = float((knn["listing_gain_pct"] > 0).mean()) if len(knn) else None

    if stage == "pre-bid":
        Xp = _frame(row, PREBID_NUM).copy()
        Xp[PREBID_NUM] = Xp[PREBID_NUM].apply(pd.to_numeric, errors="coerce")
        res["prob"] = float(models.prebid.predict_proba(Xp[PREBID_NUM])[0, 1])
        g = knn["listing_gain_pct"]
        q = {k: float(g.quantile(qq)) for qq, k in zip(QUANTILES, QKEYS)} if len(g) else {k: np.nan for k in QKEYS}
        # widen: 10 neighbours understate the spread, so stretch tails by the training-set spread
        res["wide"] = True
    else:
        X = _demand_row(row, sub)
        res["prob"] = float(models.demand.predict_proba(X[DEMAND_NUM])[0, 1])
        qib = sub.get("sub_qib_x")
        if gm == "B" and qib is not None:
            q, meta = _empirical(models.train, float(qib), pd.to_numeric(pd.Series([row.get("pat_margin_fy0_pct")]), errors="coerce").iloc[0])
            res["model_b_meta"] = meta
        else:
            q = {k: float(v[0]) for k, v in _predict_a(models.quantile_a, X).items()}
        res["wide"] = False
    q = dict(zip(QKEYS, np.sort([q[k] for k in QKEYS])))
    res["gain"] = q
    res["gain_mean"] = float(np.dot(QWEIGHTS, [q[k] for k in QKEYS]))
    res["prices"] = {k: price * (1 + q[k] / 100) if pd.notna(price) else np.nan for k in QKEYS}
    res["floor"] = stage == "provisional"
    retail = sub.get("sub_retail_x") or sub.get("sub_total_x")
    res["p_allot"] = allotment_prob(retail) if (retail and stage != "pre-bid") else float("nan")
    lot_value = lot * price if pd.notna(lot) and pd.notna(price) else APPLICATION_RS
    res["lot_value"] = lot_value
    res["profit_per_lot"] = (res["p_allot"] * res["gain_mean"] / 100 * lot_value) if pd.notna(res["p_allot"]) else float("nan")
    res["qib_needed"] = qib_needed(models, row)
    return res


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def verdict(prob: float, qib: float | None, stage: str) -> tuple[str, list[str]]:
    """Apply if P >= 0.85 and QIB >= 10x; Skip if P < 0.55 or QIB < 3x (once bidding has closed);
    Wait while subscription is unknown or still building; otherwise Apply with caution."""
    why: list[str] = []
    if stage == "pre-bid" or qib is None or pd.isna(qib):
        return "Wait for subscription", ["No subscription data yet; fundamentals alone predict listing gains only weakly."]
    meets_apply = prob >= 0.85 and qib >= 10
    if stage == "provisional":
        if meets_apply:
            return "Apply", [f"Already at {qib:.1f}x QIB with a {prob:.0%} chance, and the live figure is a floor."]
        return "Wait for subscription", [f"Bidding is still open ({qib:.1f}x QIB so far); QIBs bid mostly on the last day."]
    if meets_apply:
        return "Apply", [f"Chance of a listing gain {prob:.0%} (at least 85%).", f"QIB subscription {qib:.1f}x (at least 10x)."]
    if prob < 0.55 or qib < 3:
        if prob < 0.55:
            why.append(f"Chance of a listing gain only {prob:.0%} (below 55%).")
        if qib < 3:
            why.append(f"QIB subscription only {qib:.1f}x (below 3x).")
        return "Skip", why
    return "Apply with caution", [f"Chance {prob:.0%} is below 85%." if prob < 0.85 else f"Chance {prob:.0%}.",
                                  f"QIB subscription {qib:.1f}x." + (" Below the 10x comfort level." if qib < 10 else "")]


# --------------------------------------------------------------------------- #
# Backtest on 2026
# --------------------------------------------------------------------------- #
def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), min(bins, len(df)), labels=False, duplicates="drop") + 1
    t = df.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t


def prob_scores(p: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, p)), "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "base_rate": float(np.mean(y))}


def gain_scores(y: np.ndarray, pred: dict[str, np.ndarray]) -> dict:
    from scipy.stats import spearmanr
    ok = ~np.isnan(pred["p50"])
    y, pred = y[ok], {k: v[ok] for k, v in pred.items()}
    return {"n": int(len(y)), "mae_p50": float(np.median(np.abs(y - pred["p50"]))),
            "cov_25_75": float(np.mean((y >= pred["p25"]) & (y <= pred["p75"]))),
            "cov_10_90": float(np.mean((y >= pred["p10"]) & (y <= pred["p90"]))),
            "spearman": float(spearmanr(pred["p50"], y)[0]) if len(y) > 2 else np.nan,
            "pinball": _pinball(y, pred)}


def strategy_table(test: pd.DataFrame) -> pd.DataFrame:
    """Hit rate, average gain, allotment-weighted gain and summed expected gain per lot (Rs) by rule."""
    t = test.copy()
    t["p_allot"] = [allotment_prob(r if pd.notna(r) else tot) for r, tot in zip(t["sub_retail_x"], t["sub_total_x"])]
    lot_val = (t["lot_size"] * t["issue_price"]).where(lambda s: s > 0)
    t["lot_value"] = lot_val.fillna(lot_val.median() if lot_val.notna().any() else APPLICATION_RS)
    rules = {"Always apply": pd.Series(True, index=t.index), "Total subscription > 10x": t["sub_total_x"] > 10}
    for x in (3, 5, 10, 20, 50):
        rules[f"QIB > {x}x"] = t["sub_qib_x"] > x
    rows = []
    for name, m in rules.items():
        s = t[m.fillna(False)]
        pa = s["p_allot"].fillna(s["p_allot"].median() if s["p_allot"].notna().any() else 1.0)
        rows.append({"rule": name, "n_applied": len(s), "hit_rate": float((s["listing_gain_pct"] > 0).mean()) if len(s) else np.nan,
                     "avg_gain_pct": float(s["listing_gain_pct"].mean()) if len(s) else np.nan,
                     "allotment_weighted_gain_pct": float((pa * s["listing_gain_pct"]).mean()) if len(s) else np.nan,
                     "sum_expected_gain_per_lot_rs": float((pa * s["listing_gain_pct"] / 100 * s["lot_value"]).sum()) if len(s) else 0.0})
    return pd.DataFrame(rows).set_index("rule")


def backtest(models: Models, d: pd.DataFrame) -> dict:
    """Score the already-fitted models on 2026 IPOs. Nothing here refits anything."""
    _, test = split(d)
    res: dict = {"n_test": len(test), "n_train": models.info["n_train"], "gain_model_default": models.default_gain_model,
                 "kind_a": models.kind_a, "val_losses": models.info["pinball_val"]}
    if len(test) < 10:
        return res
    y = (test["listing_gain_pct"] > 0).astype(int).to_numpy()
    td = test[test["l_qib"].notna() | test["l_total"].notna()]
    yd = (td["listing_gain_pct"] > 0).astype(int).to_numpy()
    pd_ = models.demand.predict_proba(td[DEMAND_NUM])[:, 1]
    pp = models.prebid.predict_proba(test[PREBID_NUM])[:, 1]
    res["demand"], res["prebid"] = prob_scores(pd_, yd), prob_scores(pp, y)
    res["calibration_demand"], res["calibration_prebid"] = calibration_table(pd_, yd), calibration_table(pp, y)
    qa = _predict_a(models.quantile_a, td)
    qb_rows = [_empirical(models.train, r.sub_qib_x, r.pat_margin_fy0_pct)[0] if pd.notna(r.sub_qib_x) else {k: np.nan for k in QKEYS} for r in td.itertuples()]
    qb = {k: np.array([r[k] for r in qb_rows]) for k in QKEYS}
    yg = td["listing_gain_pct"].to_numpy()
    res["gain_a"], res["gain_b"] = gain_scores(yg, qa), gain_scores(yg, qb)
    # pre-bid gain range from nearest neighbours
    kq = []
    for _, r in test.iterrows():
        g = similar(models, r, 10)["listing_gain_pct"]
        kq.append({k: float(g.quantile(q)) if len(g) else np.nan for q, k in zip(QUANTILES, QKEYS)})
    res["gain_prebid_knn"] = gain_scores(test["listing_gain_pct"].to_numpy(), {k: np.array([r[k] for r in kq]) for k in QKEYS})
    res["strategies"] = strategy_table(test)
    # baselines for the probability models
    res["baselines"] = pd.DataFrame({
        "accuracy": [float(y.mean()), float(((test["sub_total_x"] > 10).astype(int) == y).mean())],
    }, index=["Always apply", "Total subscription > 10x"])
    return res


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
PILLARS: dict[str, list[tuple[str, int]]] = {   # metric, +1 higher is better / -1 lower is better
    "Growth": [("revenue_cagr_pct", 1), ("pat_cagr_pct", 1)],
    "Profitability": [("pat_margin_fy0_pct", 1), ("ebitda_margin_fy0_pct", 1), ("ronw_fy0_pct", 1)],
    "Balance sheet": [("debt_to_equity_fy0", -1), ("cfo_to_pat", 1)],
    "Valuation": [("pe_premium_to_peers_pct", -1), ("pe_at_issue", -1)],
    "Issue structure": [("ofs_pct", -1), ("proceeds_debt_repayment_pct", -1)],
    "Demand": [("sub_qib_x", 1), ("sub_total_x", 1)],
}


def _pct_rank(series: pd.Series, value: float, sign: int) -> float:
    s = series.dropna()
    if pd.isna(value) or len(s) < 20:
        return float("nan")
    r = float((s < value).mean() + 0.5 * (s == value).mean())
    return r if sign > 0 else 1 - r


def pillar_scores(row: pd.Series, d: pd.DataFrame) -> pd.DataFrame:
    """0-10 per pillar = mean percentile rank vs every past IPO; plus how well the pillar predicted
    listing results on the training rows (Spearman with the gain, AUC for gain vs not)."""
    hist = d[d["listing_gain_pct"].notna()]
    train, _ = split(d)
    rows = []
    for name, metrics in PILLARS.items():
        ranks = [_pct_rank(hist[c], pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0], s) for c, s in metrics if c in hist]
        ranks = [r for r in ranks if not np.isnan(r)]
        score = 10 * float(np.mean(ranks)) if ranks else float("nan")
        per = []
        for c, s in metrics:
            if c in train:
                per.append(train[c].rank(pct=True) if s > 0 else 1 - train[c].rank(pct=True))
        ps = pd.concat(per, axis=1).mean(axis=1) if per else pd.Series(dtype=float)
        ok = ps.notna()
        sp = auc = float("nan")
        if ok.sum() > 30:
            from scipy.stats import spearmanr
            sp = float(spearmanr(ps[ok], train.loc[ok, "listing_gain_pct"])[0])
            yb = (train.loc[ok, "listing_gain_pct"] > 0).astype(int)
            auc = float(roc_auc_score(yb, ps[ok])) if yb.nunique() == 2 else float("nan")
        rows.append({"pillar": name, "score_0_10": score, "metrics_used": len(ranks), "train_spearman_vs_gain": sp, "train_auc": auc})
    return pd.DataFrame(rows).set_index("pillar")


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    company: str
    symbol: str
    est: dict
    verdict: str
    reasons: list[str]
    sub: dict
    closed: bool
    band_table: pd.DataFrame
    key_numbers: list[tuple[str, str, str]]       # label, value, note/page
    financials: pd.DataFrame
    peers: pd.DataFrame
    proceeds: pd.DataFrame
    sellers: pd.DataFrame
    scorecard: pd.DataFrame
    flags: list
    qual: dict | None
    similar: pd.DataFrame
    backtest: dict
    notes: list[str] = field(default_factory=list)


def _f(v, fmt="{:,.1f}", na="–"):
    return na if v is None or pd.isna(v) else fmt.format(v)


def _num(v) -> float:
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]


def band_table(d: pd.DataFrame) -> pd.DataFrame:
    """How past IPOs listed at each final total-subscription band (training rows)."""
    train, _ = split(d)
    labels = ["under 3x", "3-10x", "10-30x", "30-100x", "over 100x"]
    t = train[train["sub_total_x"].notna()].copy()
    t["band"] = pd.cut(t["sub_total_x"], [0, 3, 10, 30, 100, 1e9], labels=labels)
    g = t.groupby("band", observed=True)["listing_gain_pct"]
    return pd.DataFrame({"IPOs": g.size(), "listed above issue": g.apply(lambda s: (s > 0).mean()),
                         "median gain %": g.median(), "p25 %": g.quantile(.25), "p75 %": g.quantile(.75)}).reindex(labels)


def _pg(record: dict | None, field_name: str):
    return (record or {}).get("fields", {}).get(f"rhp_page_{field_name}")


def key_numbers(row: pd.Series, record: dict | None) -> list[tuple[str, str, str]]:
    g = lambda c: _num(row.get(c))
    pg = lambda c: f"PDF p.{int(_pg(record, c))}" if _pg(record, c) else ""
    peer_note = f"peer median {_f(g('peer_pe_median'), '{:.1f}')}"
    if pd.notna(g("pe_premium_to_peers_pct")):
        peer_note += f" ({_f(g('pe_premium_to_peers_pct'), '{:+.0f}')}% vs peers)"
    return [
        ("Market cap at top of band", f"₹{_f(g('mcap_at_issue_cr'), '{:,.0f}')} cr", ""),
        ("Issue size", f"₹{_f(g('issue_size_cr'), '{:,.0f}')} cr", ""),
        ("Fresh issue / offer for sale", f"₹{_f(0.0 if pd.isna(g('fresh_issue_cr')) and g('ofs_pct') == 100 else g('fresh_issue_cr'), '{:,.0f}')} cr / ₹{_f(g('ofs_cr'), '{:,.0f}')} cr ({_f(g('ofs_pct'), '{:.0f}')}% OFS)", pg("fresh_issue_cr")),
        ("Promoter holding before → after", f"{_f(g('promoter_pre_pct'), '{:.1f}')}% → {_f(g('promoter_post_pct'), '{:.1f}')}%", pg("promoter_pre_pct")),
        ("P/E at top of band", _f(g("pe_at_issue"), "{:.1f}"), peer_note),
        ("P/B at top of band", _f(g("pb_at_issue"), "{:.1f}"), "peer P/B not extracted"),
        ("P/S at top of band", _f(g("ps_at_issue"), "{:.1f}"), "peer P/S not extracted"),
        ("PAT margin / EBITDA margin (latest year)", f"{_f(g('pat_margin_fy0_pct'))}% / {_f(g('ebitda_margin_fy0_pct'))}%", ""),
        ("Return on net worth", f"{_f(g('ronw_fy0_pct'))}%", pg("ronw_fy0_pct")),
        ("Debt / equity", _f(g("debt_to_equity_fy0"), "{:.2f}"), ""),
        ("Operating cash flow / profit", _f(g("cfo_to_pat"), "{:.2f}"), ""),
        ("Pre-IPO placement price", f"₹{_f(g('pre_ipo_price'), '{:,.1f}')}" if pd.notna(g("pre_ipo_price")) else "none found",
         f"IPO price {_f(g('ipo_premium_to_preipo_pct'), '{:+.0f}')}% vs that" if pd.notna(g("ipo_premium_to_preipo_pct")) else ""),
        ("Anchor portion", f"{_f(g('anchor_pct_of_issue'), '{:.0f}')}% of issue" if pd.notna(g("anchor_pct_of_issue")) else "not published yet", ""),
    ]


def financial_table(row: pd.Series, record: dict | None) -> pd.DataFrame:
    fy = (record or {}).get("extract", {}).get("fy0_label") or "latest FY"
    cols = {"FY-2": "fy2", "FY-1": "fy1", f"FY0 ({fy})": "fy0"}
    names = {"revenue": "Revenue", "ebitda": "EBITDA", "pat": "PAT", "cfo": "Operating cash flow"}
    t = pd.DataFrame({lab: {names[m]: _num(row.get(f"{m}_{suf}")) for m in names} for lab, suf in cols.items()})
    if pd.notna(_num(row.get("revenue_stub"))):
        months = _num(row.get("stub_period_months"))
        t[f"Stub ({int(months)}m)" if pd.notna(months) else "Stub"] = [_num(row.get("revenue_stub")), np.nan, _num(row.get("pat_stub")), np.nan]
    t.index.name = "₹ crore"
    return t


def peer_table(record: dict | None, row: pd.Series) -> pd.DataFrame:
    peers = (record or {}).get("extract", {}).get("peers") or []
    t = pd.DataFrame([{"peer": p["name"], "P/E": p.get("pe"), "RoNW %": p.get("ronw_pct"), "PDF page": p.get("page")} for p in peers])
    me = pd.DataFrame([{"peer": f"{row.get('company')} (this IPO)", "P/E": row.get("pe_at_issue"), "RoNW %": row.get("ronw_fy0_pct"), "PDF page": None}])
    return pd.concat([me, t], ignore_index=True) if len(t) else me


def proceeds_table(record: dict | None) -> pd.DataFrame:
    from . import extract_rhp as X
    rows = [{"use of fresh money": o["purpose"], "type": o["category"], "₹ crore": X.crore(X.Fig(**o["amount"])), "PDF page": o["amount"].get("page")}
            for o in (record or {}).get("extract", {}).get("objects_of_offer", []) or []]
    return pd.DataFrame(rows)


def sellers_table(record: dict | None) -> pd.DataFrame:
    from . import extract_rhp as X
    rows = [{"seller": s["name"], "type": s["kind"], "shares offered": X.absolute(X.Fig(**s["shares_offered"])), "PDF page": s["shares_offered"].get("page")}
            for s in (record or {}).get("extract", {}).get("selling_shareholders", []) or []]
    return pd.DataFrame(rows)


def build_report(row: pd.Series, record: dict | None, qual_rec: dict | None, models: Models, d: pd.DataFrame,
                 sub: dict | None = None, closed: bool = False, gain_model: str | None = None, backtest_res: dict | None = None) -> Report:
    from . import live
    row = row.copy()
    for c, v in (sub or {}).items():          # scorecard and similar-IPO search should see the subscription too
        if v is not None and pd.notna(v) and v > 0:
            row[c] = float(v)
    est = estimate(models, row, sub, closed, gain_model)
    qib = (sub or {}).get("sub_qib_x")
    label, reasons = verdict(est["prob"], qib, est["stage"])
    if est["stage"] != "pre-bid":
        reasons = reasons + [f"Model {est['gain_model']}: median expected listing gain {est['gain']['p50']:+.0f}%."]
    elif est["knn_hit_rate"] is not None:
        reasons = reasons + [f"The 10 most similar past IPOs listed above issue {est['knn_hit_rate']:.0%} of the time."]
    flags = live.flags(row, d[d["listing_gain_pct"].notna()])
    return Report(
        company=str(row.get("company")), symbol=str(row.get("symbol")), est=est, verdict=label, reasons=reasons[:5],
        sub=dict(sub or {}), closed=closed, band_table=band_table(d), key_numbers=key_numbers(row, record),
        financials=financial_table(row, record), peers=peer_table(record, row), proceeds=proceeds_table(record),
        sellers=sellers_table(record), scorecard=pillar_scores(row, d), flags=flags, qual=(qual_rec or {}).get("qual"),
        similar=est["similar"], backtest=backtest_res or {})


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.2f}"
    return str(v)


def _md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    if df is None or df.empty:
        return "_none found_\n"
    t = df.copy()
    fmt = dict(fmt or {})
    for c in t.columns:
        if "page" in str(c).lower():
            fmt.setdefault(c, lambda v: "–" if v is None or pd.isna(v) else f"{int(v)}")
        elif "shares" in str(c).lower():
            fmt.setdefault(c, lambda v: "–" if v is None or pd.isna(v) else f"{v:,.0f}")
    for c in t.columns:
        f = fmt.get(c)
        t[c] = [f(v) if f else _cell(v) for v in t[c]]
    return t.to_markdown() + "\n"


def _pt(p: dict) -> str:
    return f"{p['text']} _(PDF p.{p['page']})_" if p.get("page") else p["text"]


def to_markdown(r: Report) -> str:
    e, g = r.est, r.est["gain"]
    L: list[str] = [f"# {r.company} ({r.symbol}): IPO report", "",
                    "_Historical patterns only; not investment advice. Models fitted on IPOs listed 2022-2025; scores below are on 2026._", ""]
    L += ["## 1. Verdict", f"**{r.verdict}**", ""] + [f"- {x}" for x in r.reasons] + [""]
    L += ["## 2. Listing chance and expected gain",
          f"Stage: **{e['stage']}**" + (" (live figures are a floor: QIBs bid mostly on the last day)" if e["floor"] else ""), "",
          f"- Chance of listing above issue price: **{e['prob']:.0%}**" + (" (at least)" if e["floor"] else ""),
          f"- Expected listing gain: **{g['p50']:+.0f}%** median (likely range {g['p25']:+.0f}% to {g['p75']:+.0f}%; 80% range {g['p10']:+.0f}% to {g['p90']:+.0f}%)"
          + (" - wide, pre-bid estimate" if e.get("wide") else ""),
          f"- Implied listing price: ₹{_f(e['prices']['p50'], '{:,.0f}')} (₹{_f(e['prices']['p25'], '{:,.0f}')} to ₹{_f(e['prices']['p75'], '{:,.0f}')}; "
          f"80% range ₹{_f(e['prices']['p10'], '{:,.0f}')} to ₹{_f(e['prices']['p90'], '{:,.0f}')}) on a top-of-band price of ₹{_f(e['price'], '{:,.0f}')}",
          f"- Allotment odds per retail application: {_f(e['p_allot'], '{:.0%}')}; expected profit per lot: ₹{_f(e['profit_per_lot'], '{:,.0f}')} (lot value ₹{_f(e['lot_value'], '{:,.0f}')})",
          "- QIB subscription needed for a 70% / 85% / 95% chance: " + " / ".join(_f(e['qib_needed'][t], "{:.0f}x", ">3000x") for t in (0.70, 0.85, 0.95)), ""]
    bt = r.backtest
    if bt.get("demand"):
        L += [f"Scores on {bt['n_test']} IPOs from 2026: subscription model AUC {bt['demand']['auc']:.2f}, accuracy {bt['demand']['accuracy']:.0%}; "
              f"fundamentals-only AUC {bt['prebid']['auc']:.2f}; gain median absolute error {bt['gain_a']['mae_p50']:.1f} points (Model A), "
              f"25-75 range covers {bt['gain_a']['cov_25_75']:.0%}, 10-90 covers {bt['gain_a']['cov_10_90']:.0%}.", ""]
    L += ["## 3. Subscription"]
    cats = [("sub_total_x", "Total"), ("sub_qib_x", "QIB"), ("sub_nii_x", "NII"), ("sub_retail_x", "Retail")]
    L += ["| " + " | ".join(n for _, n in cats) + " |", "|" + "---|" * len(cats),
          "| " + " | ".join(_f(r.sub.get(c), "{:.2f}x") for c, _ in cats) + " |", "",
          "How past IPOs listed by final total subscription:", "", _md_table(r.band_table, {"listed above issue": lambda v: _f(v, "{:.0%}")})]
    q = r.qual or {}
    L += ["## 4. The business"]
    if q:
        for key, title in (("business_summary", "What it does"), ("industry", "Industry")):
            L += [f"**{title}**"] + [f"- {_pt(p)}" for p in q.get(key, [])] + [""]
        L += ["**Segments**"] + [f"- {s['name']}" + (f": {s['share_of_revenue']}" if s.get("share_of_revenue") else "") + (f" _(p.{s['page']})_" if s.get("page") else "") for s in q.get("segments", [])] + [""]
        L += ["**Concentration**"] + [f"- {_pt(c)}" for c in q.get("concentration", [])] + [""]
        L += ["**Promoters**"] + [f"- {p['name']}: {p['background']}" + (f" _(p.{p['page']})_" if p.get("page") else "") for p in q.get("promoters", [])] + [""]
    else:
        L += ["_Qualitative pass not run._", ""]
    L += ["## 5. Key numbers", "| Item | Value | Note |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in r.key_numbers] + [""]
    L += ["**Financials**", "", _md_table(r.financials.reset_index()), "**Peers**", "", _md_table(r.peers),
          "**Use of fresh money**", "", _md_table(r.proceeds), "**Selling shareholders**", "", _md_table(r.sellers)]
    L += ["## 6. Scorecard (percentile rank vs all past IPOs, 0-10)", "", _md_table(r.scorecard.reset_index()),
          "_train_spearman_vs_gain and train_auc show how well each pillar predicted listing results on 2022-2025._", ""]
    L += ["## 7. Strengths, concerns and risks", "**Strengths**"] + [f"- {f.text}" + (f" ({f.evidence})" if f.evidence else "") for f in r.flags if f.kind == "good"]
    L += [f"- {_pt(p)}" for p in q.get("strengths", [])]
    L += ["", "**Concerns**"] + [f"- {f.text}" + (f" ({f.evidence})" if f.evidence else "") for f in r.flags if f.kind == "bad"]
    if q:
        L += ["", "**Top risks**"] + [f"- [{x['severity']}] {x['title']}: {x['detail']}" + (f" _(p.{x['page']})_" if x.get("page") else "") for x in q.get("risks", [])]
        L += ["", "**Litigation**"] + [f"- {x['category']}: {x.get('count') or '?'} cases, {x.get('amount') or 'amount not stated'}" + (f" _(p.{x['page']})_" if x.get("page") else "") for x in q.get("litigation", [])]
        L += ["", "**Contingent liabilities**"] + [f"- {_pt(p)}" for p in q.get("contingent_liabilities", [])]
        L += ["", "**Related-party transactions**"] + [f"- {_pt(p)}" for p in q.get("related_party", [])]
        L += ["", "**Red flags**"] + [f"- {_pt(p)}" for p in q.get("red_flags", [])]
    sim = r.similar.assign(listing_date=r.similar["listing_date"].dt.strftime("%Y-%m-%d")) if len(r.similar) else r.similar
    L += ["", "## 8. Most similar past IPOs", "", _md_table(sim), ""]
    L += [f"> {n}" for n in r.notes]
    return "\n".join(L)
