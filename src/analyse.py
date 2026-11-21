"""Winners vs losers, simple rules, logistic regression, and the out-of-sample 2026 test.

Everything that is *fitted or chosen* (thresholds, rules, model coefficients, which
features to use) only ever sees the training rows. The test rows are touched once, in
`evaluate_all`, to report results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_FEATURES, FEATURES, TEST_START, TRAIN_START

TARGETS = ("target_gain", "target_gain_5")


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["issue_open", "issue_close", "listing_date"])
    return add_split(df)


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    ld = pd.to_datetime(df["listing_date"], errors="coerce")
    split = np.where(ld >= pd.Timestamp(TEST_START), "test",
                     np.where(ld >= pd.Timestamp(TRAIN_START), "train", "warmup"))
    return df.assign(split=split)


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURES).copy()
    if "pre_ipo_placement" in X:
        X["pre_ipo_placement"] = X["pre_ipo_placement"].map({"yes": 1, "no": 0, 1: 1, 0: 0})
    return X.apply(pd.to_numeric, errors="coerce")


def usable_features(train: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Features filled in for at least `min_coverage` of training rows and not constant."""
    X = numeric_features(train)
    cov = X.notna().mean()
    return [c for c in X.columns if cov[c] >= min_coverage and X[c].nunique(dropna=True) > 1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, gains=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    out = {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "accuracy": (tp + tn) / n if n else np.nan,
           "precision": tp / (tp + fp) if tp + fp else np.nan,
           "recall": tp / (tp + fn) if tp + fn else np.nan,
           "n_apply": tp + fp}
    if gains is not None:
        g = np.asarray(gains, dtype=float)
        picked = g[y_pred == 1]
        out["avg_gain_if_applied_pct"] = float(np.nanmean(picked)) if len(picked) else np.nan
        out["avg_gain_all_pct"] = float(np.nanmean(g)) if len(g) else np.nan
    return out


def base_rate(df: pd.DataFrame, target: str = "target_gain") -> dict:
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    return {"n": int(len(y)), "positive": int(y.sum()), "rate": float(y.mean()) if len(y) else np.nan}


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_always(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=int)


def baseline_sub10(df: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(df["sub_total_x"], errors="coerce") > 10).fillna(False).astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# Winners vs losers
# --------------------------------------------------------------------------- #
def compare_groups(train: pd.DataFrame, target: str = "target_gain", features: list[str] | None = None) -> pd.DataFrame:
    """Per feature: median in each group, and share above the overall training median."""
    from scipy.stats import mannwhitneyu

    y = pd.to_numeric(train[target], errors="coerce")
    X = numeric_features(train)
    rows = []
    for f in features or usable_features(train, 0.3):
        x = X[f]
        ok = x.notna() & y.notna()
        gain, loss = x[ok & (y == 1)], x[ok & (y == 0)]
        if len(gain) < 5 or len(loss) < 5:
            continue
        thr = float(x[ok].median())
        try:
            pval = float(mannwhitneyu(gain, loss).pvalue)
        except ValueError:
            pval = np.nan
        rows.append({"feature": f, "n_gain": len(gain), "n_loss": len(loss),
                     "median_gain": float(gain.median()), "median_loss": float(loss.median()),
                     "threshold": thr,
                     "share_above_gain": float((gain > thr).mean() * 100),
                     "share_above_loss": float((loss > thr).mean() * 100),
                     "p_value": pval})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["share_gap_pp"] = out["share_above_gain"] - out["share_above_loss"]
    return out.sort_values("share_gap_pp", key=abs, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    feature: str
    op: str          # ">=" or "<="
    threshold: float

    def mask(self, df: pd.DataFrame, X: pd.DataFrame | None = None) -> np.ndarray:
        """X = numeric_features(df), passed in when evaluating many conditions on the same rows."""
        x = (numeric_features(df) if X is None else X)[self.feature]
        m = x >= self.threshold if self.op == ">=" else x <= self.threshold
        return m.fillna(False).to_numpy()   # missing value -> condition not met

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.4g}"


@dataclass(frozen=True)
class RuleSet:
    conditions: tuple[Condition, ...]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        X = numeric_features(df)
        for c in self.conditions:
            m &= c.mask(df, X)
        return m.astype(int)

    def __str__(self) -> str:
        return " AND ".join(map(str, self.conditions)) or "(always apply)"


def single_conditions(train: pd.DataFrame, target: str = "target_gain", min_coverage: float = 0.2,
                      features: list[str] | None = None) -> pd.DataFrame:
    """Best threshold per feature by training precision, among cut-offs that still select
    at least `min_coverage` of the training IPOs."""
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).to_numpy().astype(int)
    X = numeric_features(train)
    rows = []
    for f in features or usable_features(train):
        x = X[f]
        qs = np.unique(np.nanquantile(x.dropna(), np.linspace(0.1, 0.9, 17))) if x.notna().sum() >= 20 else []
        best = None
        for thr in qs:
            for op in (">=", "<="):
                c = Condition(f, op, float(thr))
                pred = c.mask(train, X)
                if pred.mean() < min_coverage:
                    continue
                prec = y[pred].mean()
                if best is None or prec > best[0]:
                    best = (prec, pred.mean(), c)
        if best:
            rows.append({"condition": best[2], "feature": f, "train_precision": best[0], "train_coverage": best[1]})
    out = pd.DataFrame(rows)
    return out.sort_values("train_precision", ascending=False).reset_index(drop=True) if len(out) else out


def greedy_rules(train: pd.DataFrame, target: str = "target_gain", max_conditions: int = 3,
                 min_coverage: float = 0.25) -> RuleSet:
    """Add conditions (one per feature) while training precision improves and coverage stays above the floor."""
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).to_numpy().astype(int)
    cands = single_conditions(train, target, min_coverage=min_coverage)
    chosen: list[Condition] = []
    X = numeric_features(train)
    mask = np.ones(len(train), dtype=bool)
    best_prec = y.mean() if len(y) else 0
    for _ in range(max_conditions):
        pick = None
        for c in cands["condition"] if len(cands) else []:
            if c.feature in {k.feature for k in chosen}:
                continue
            m = mask & c.mask(train, X)
            if m.mean() < min_coverage or not m.any():
                continue
            prec = y[m].mean()
            if prec > best_prec + 0.01 and (pick is None or prec > pick[0]):
                pick = (prec, c, m)
        if pick is None:
            break
        best_prec, c, mask = pick
        chosen.append(c)
    return RuleSet(tuple(chosen))


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #
def fit_logistic(train: pd.DataFrame, target: str = "target_gain", features: list[str] | None = None, C: float = 1.0):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    feats = features or usable_features(train)
    cats = [c for c in CATEGORICAL_FEATURES if c in train and train[c].notna().mean() >= 0.5]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), feats),
        *([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5), cats)] if cats else []),
    ])
    model = Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=2000))])
    X = _model_frame(train, feats, cats)
    y = pd.to_numeric(train[target], errors="coerce")
    ok = y.notna().to_numpy()
    model.fit(X[ok], y[ok].astype(int))
    model.feats_, model.cats_ = feats, cats
    return model


def _model_frame(df, feats, cats):
    X = numeric_features(df)[feats]
    for c in cats:
        X[c] = df[c].fillna("Unknown").astype(str)
    return X


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_model_frame(df, model.feats_, model.cats_))[:, 1]


def coefficients(model) -> pd.DataFrame:
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["lr"].coef_[0]
    out = pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names], "coef": coef})
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #
def evaluate_all(df: pd.DataFrame, target: str = "target_gain", quality: tuple[str, ...] = ("full", "partial")) -> dict:
    df = add_split(df) if "split" not in df else df
    df = df[df["data_quality"].isin(quality) | df["data_quality"].isna()] if "data_quality" in df else df
    df = df[pd.to_numeric(df[target], errors="coerce").notna()]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    res = {"target": target, "base_rate_train": base_rate(train, target), "base_rate_test": base_rate(test, target),
           "groups": compare_groups(train, target) if len(train) else pd.DataFrame()}
    if len(train) < 30:
        res["error"] = f"only {len(train)} usable training rows; need at least 30"
        return res

    rules = greedy_rules(train, target)
    model = fit_logistic(train, target)
    res["single_conditions"] = single_conditions(train, target).head(15)
    res["rules"] = rules
    res["coefficients"] = coefficients(model)

    def score(part: pd.DataFrame) -> pd.DataFrame:
        y, g = part[target].astype(int), part["listing_gain_pct"]
        preds = {"Always apply": baseline_always(part),
                 "Subscribed > 10x": baseline_sub10(part),
                 f"Rules: {rules}": rules.predict(part),
                 "Logistic (p > 0.5)": (predict_proba(model, part) > 0.5).astype(int) if len(part) else np.array([], int)}
        return pd.DataFrame({k: metrics(y, v, g) for k, v in preds.items()}).T

    res["train_scores"] = score(train)
    res["test_scores"] = score(test) if len(test) else pd.DataFrame()
    res["model"] = model
    return res


def report_markdown(res: dict) -> str:
    def tbl(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
        if df is None or len(df) == 0:
            return "_none_\n"
        return df.to_markdown(floatfmt=floatfmt) + "\n"

    br, bt = res["base_rate_train"], res["base_rate_test"]
    lines = [f"# IPO listing-day analysis — target `{res['target']}`", "",
             f"**Base rate (train 2022–2025):** {br['positive']} of {br['n']} IPOs = {br['rate'] * 100:.1f}%",
             f"**Base rate (test 2026):** {bt['positive']} of {bt['n']} IPOs = "
             + (f"{bt['rate'] * 100:.1f}%" if bt["n"] else "n/a"), ""]
    if "error" in res:
        return "\n".join(lines + [f"**Stopped:** {res['error']}"])
    lines += ["## Gain vs loss groups (training rows)", tbl(res["groups"]),
              "## Best single conditions (training)", tbl(res["single_conditions"].drop(columns="condition", errors="ignore")),
              f"## Rule set chosen on training data\n\n`{res['rules']}`\n",
              "## Logistic regression coefficients (standardised)", tbl(res["coefficients"].head(20)),
              "## Training scores", tbl(res["train_scores"]),
              "## Test scores (2026, out of sample)", tbl(res["test_scores"])]
    return "\n".join(lines)
