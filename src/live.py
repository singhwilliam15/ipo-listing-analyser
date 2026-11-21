"""Analyse an IPO that has not listed yet.

    find_issue("mahanadi")            -> match on NSE's current / upcoming issues
    analyse_issue(match, pdf=None)    -> RHP download + Claude extraction + flags + scenarios

Everything here is a *historical-pattern* estimate built from 2022-2026 mainboard IPOs. It is
not investment advice, and before bidding closes the evidence is weak (see `PRE_BID_NOTE`).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import analyse, extract_rhp, features, schema
from .fetch_market import DataError, NSEClient, find_rhp_url, index_closes, issue_info, parse_bids

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PRE_BID_NOTE = ("Before bidding closes the only inputs are fundamentals and market mood. On 2022-2026 data they "
                "separate gainers from losers only weakly (AUC about 0.65 on 2026 IPOs, where 0.5 is a coin flip). "
                "Final subscription is far more informative, so treat early numbers as a rough prior.")

MODEL_FEATURES = ["revenue_cagr_pct", "pat_margin_fy0_pct", "ronw_fy0_pct", "debt_to_equity_fy0", "pe_at_issue",
                  "pe_premium_to_peers_pct", "ofs_pct", "log_issue_size", "loss_making_flag",
                  "proceeds_debt_repayment_pct", "nifty_ret_1m_pct", "avg_listing_gain_prev_5_pct",
                  "ipos_listed_prev_30d", "india_vix_close", "promoter_pre_pct"]

SUB_BUCKETS = [0, 3, 10, 30, 100, 1e9]
SUB_LABELS = ["under 3x", "3-10x", "10-30x", "30-100x", "over 100x"]


# --------------------------------------------------------------------------- #
# Finding the issue
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    symbol: str
    company: str
    status: str                 # Active / Forthcoming / Listed
    segment: str
    price_low: float | None = None
    price_high: float | None = None
    issue_open: str | None = None
    issue_close: str | None = None
    score: float = 0.0


def _num(s) -> float | None:
    m = re.findall(r"\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m[-1]) if m else None


def list_open_issues(nse: NSEClient | None = None) -> pd.DataFrame:
    """Current and forthcoming issues from NSE (mainboard and SME)."""
    nse = nse or NSEClient()
    rows = []
    for path, kw in (("ipo-current-issue", {}), ("all-upcoming-issues", {"category": "ipo"})):
        try:
            js = nse.get_json(path, **kw)
        except DataError:
            continue
        rows += js if isinstance(js, list) else js.get("data", [])
    out = []
    for r in rows:
        if not r.get("symbol"):
            continue
        band = str(r.get("issuePrice") or "")
        nums = re.findall(r"\d+(?:\.\d+)?", band.replace(",", ""))
        out.append({"symbol": r["symbol"], "company": r.get("companyName"), "status": r.get("status"),
                    "segment": "SME" if str(r.get("series")).upper() in ("SME", "ST") or r.get("segment") == "SME" else "Mainboard",
                    "price_low": float(nums[0]) if nums else None, "price_high": float(nums[-1]) if nums else None,
                    "issue_open": r.get("issueStartDate"), "issue_close": r.get("issueEndDate")})
    df = pd.DataFrame(out)
    return df.drop_duplicates("symbol") if len(df) else df


def find_issue(query: str, nse: NSEClient | None = None, include_past: bool = True, limit: int = 5) -> list[Match]:
    """Best fuzzy matches for a company name or NSE symbol among open issues (and past ones, for testing)."""
    q = re.sub(r"\b(ltd|limited|ipo|india|pvt|private)\b", "", query.lower()).strip()
    cands = list_open_issues(nse)
    cands = cands.to_dict("records") if len(cands) else []
    if include_past and (DATA / "issues.csv").exists():
        past = pd.read_csv(DATA / "issues.csv")
        past = past[past["segment"] == "Mainboard"]
        cands += [{"symbol": r.symbol, "company": r.company, "status": "Listed", "segment": "Mainboard",
                   "price_low": r.price_low, "price_high": r.price_high, "issue_open": None, "issue_close": str(r.issue_close)[:10]}
                  for r in past.itertuples()]
    scored = []
    for c in cands:
        name = re.sub(r"\b(ltd|limited|india|pvt|private)\b", "", str(c["company"]).lower()).strip()
        sym = str(c["symbol"]).lower()
        s = max(difflib.SequenceMatcher(None, q, name).ratio(), difflib.SequenceMatcher(None, q, sym).ratio())
        if q and (q in name or q in sym):
            s = max(s, 0.9)
        scored.append(Match(score=s, **c))
    scored = [m for m in sorted(scored, key=lambda m: -m.score) if m.score >= 0.45]
    seen, out = set(), []
    for m in scored:
        if m.symbol not in seen:
            seen.add(m.symbol)
            out.append(m)
    return out[:limit]


# --------------------------------------------------------------------------- #
# History used for comparisons
# --------------------------------------------------------------------------- #
def load_history(path: Path | None = None) -> pd.DataFrame:
    df = analyse.add_split(analyse.load(path or DATA / "ipos.csv"))
    df = df[df["listing_gain_pct"].notna()].copy()
    df["log_issue_size"] = np.log1p(df["issue_size_cr"])
    df["log_sub"] = np.log1p(df["sub_total_x"])
    return df


def subscription_table(hist: pd.DataFrame) -> pd.DataFrame:
    """For each final-subscription bucket: how often IPOs listed above issue and how big the move was."""
    h = hist[hist["sub_total_x"].notna()].copy()
    h["bucket"] = pd.cut(h["sub_total_x"], SUB_BUCKETS, labels=SUB_LABELS)
    g = h.groupby("bucket", observed=True)["listing_gain_pct"]
    t = pd.DataFrame({"n": g.size(), "gain_rate": g.apply(lambda s: (s > 0).mean()),
                      "p25": g.quantile(.25), "median": g.median(), "p75": g.quantile(.75)})
    return t.reindex(SUB_LABELS)


def _pipe(model):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def fit_prior_models(hist: pd.DataFrame) -> dict:
    """Fundamentals-only models: P(listing gain) and final subscription. Deliberately small and
    heavily regularised; the 2026 score is reported so the page can show how weak they are."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    h = hist[hist["data_quality"].isin(["full", "partial"])]
    tr, te = h[h["split"] == "train"], h[h["split"] == "test"]
    clf = _pipe(LogisticRegression(C=0.05, max_iter=1000)).fit(tr[MODEL_FEATURES], tr["target_gain"])
    reg = _pipe(Ridge(alpha=100)).fit(tr[MODEL_FEATURES], tr["log_sub"])
    res = {"clf": clf, "reg": reg, "auc": np.nan, "r2": np.nan, "n_train": len(tr), "n_test": len(te)}
    if len(te) > 10 and te["target_gain"].nunique() == 2:
        res["auc"] = float(roc_auc_score(te["target_gain"], clf.predict_proba(te[MODEL_FEATURES])[:, 1]))
        res["r2"] = float(r2_score(te["log_sub"], reg.predict(te[MODEL_FEATURES])))
    # refit on everything so the live prediction uses all history
    clf.fit(h[MODEL_FEATURES], h["target_gain"])
    reg.fit(h[MODEL_FEATURES], h["log_sub"])
    res["resid_sd"] = float(np.std(h["log_sub"] - reg.predict(h[MODEL_FEATURES])))
    return res


# --------------------------------------------------------------------------- #
# Strengths and weaknesses
# --------------------------------------------------------------------------- #
@dataclass
class Flag:
    kind: str           # "good" | "bad" | "info"
    text: str
    evidence: str = ""


def _rate(hist: pd.DataFrame, mask: pd.Series) -> str:
    sub = hist[mask.fillna(False)]
    if len(sub) < 8:
        return ""
    return f"historically {(sub['listing_gain_pct'] > 0).mean():.0%} of {len(sub)} similar IPOs listed above issue (median {sub['listing_gain_pct'].median():+.1f}%)"


def flags(row: pd.Series, hist: pd.DataFrame) -> list[Flag]:
    """Plain-language good / bad points, each with how similar IPOs actually listed."""
    out: list[Flag] = []
    g = lambda c: row.get(c) if c in row and pd.notna(row.get(c)) else None
    de, lm = g("debt_to_equity_fy0"), g("loss_making_flag")
    if lm == 1:
        out.append(Flag("bad", "Loss-making (or negative PAT in the latest year)", _rate(hist, hist["loss_making_flag"] == 1)))
    elif g("pat_fy0") is not None:
        out.append(Flag("good", "Profitable in the latest year", ""))
    if de is not None:
        if de > 2:
            out.append(Flag("bad", f"High leverage: debt/equity {de:.1f}", _rate(hist, hist["debt_to_equity_fy0"] > 2)))
        elif de <= 1:
            out.append(Flag("good", f"Moderate leverage: debt/equity {de:.2f}", _rate(hist, hist["debt_to_equity_fy0"] <= 1)))
    for col, label, good, bad in (("revenue_cagr_pct", "Revenue CAGR", 25, 10), ("ronw_fy0_pct", "Return on net worth", 20, 10),
                                  ("pat_margin_fy0_pct", "PAT margin", 9, 4)):
        v = g(col)
        if v is None:
            continue
        if v >= good:
            out.append(Flag("good", f"{label} {v:.1f}% (strong)", _rate(hist, hist[col] >= good)))
        elif v < bad:
            out.append(Flag("bad", f"{label} {v:.1f}% (weak)", _rate(hist, hist[col] < bad)))
    prem = g("pe_premium_to_peers_pct")
    if prem is not None:
        if prem <= -20:
            out.append(Flag("good", f"Priced {abs(prem):.0f}% below peer median P/E", _rate(hist, hist["pe_premium_to_peers_pct"] <= -20)))
        elif prem >= 50:
            out.append(Flag("bad", f"Priced {prem:.0f}% above peer median P/E", _rate(hist, hist["pe_premium_to_peers_pct"] >= 50)))
    elif g("pe_at_issue") is not None:
        out.append(Flag("info", f"P/E at issue {row['pe_at_issue']:.1f}; peer P/E not extracted", ""))
    dr = g("proceeds_debt_repayment_pct")
    if dr is not None and dr >= 50:
        out.append(Flag("bad", f"{dr:.0f}% of fresh proceeds go to debt repayment", _rate(hist, hist["proceeds_debt_repayment_pct"] >= 50)))
    ofs = g("ofs_pct")
    if ofs is not None:
        if ofs >= 99:
            out.append(Flag("info", "Entire issue is an offer for sale: existing holders sell, the company gets no money",
                            "analysts often criticise this, but OFS share alone did not separate winners from losers in 2022-2026 data"))
        elif ofs == 0:
            out.append(Flag("good", "Pure fresh issue: all money goes to the company", ""))
    pm = g("ipo_premium_to_preipo_pct")
    if pm is not None and pm > 0:
        out.append(Flag("good", f"IPO price is {pm:.0f}% above the pre-IPO placement price", _rate(hist, hist["ipo_premium_to_preipo_pct"] > 0)))
    elif pm is not None and pm < 0:
        out.append(Flag("bad", f"IPO price is {abs(pm):.0f}% below the pre-IPO placement price", ""))
    cfo = g("cfo_to_pat")
    if cfo is not None and cfo < 0.4 and (g("pat_fy0") or 0) > 0:
        out.append(Flag("bad", f"Weak cash conversion: operating cash flow is only {cfo:.0%} of profit", ""))
    if g("pat_jump_flag") == 1:
        out.append(Flag("bad", "Profit jumped sharply just before the IPO; check whether it is sustainable", ""))
    for c in (row.get("rhp_checks") or []):
        out.append(Flag("info", f"Extraction check: {c}", ""))
    return out


# --------------------------------------------------------------------------- #
# Putting one IPO together
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    match: Match
    row: pd.Series
    flags: list[Flag]
    live_sub: dict
    prior_gain_prob: float | None
    prior_sub_median: float | None
    prior_sub_range: tuple[float, float] | None
    scenarios: pd.DataFrame
    model_quality: dict
    sub_table: pd.DataFrame
    source: str                 # which document the numbers came from
    notes: list[str] = field(default_factory=list)


def get_document(symbol: str, detail, pdf: Path | None, nse: NSEClient) -> tuple[Path, str]:
    """Return (pdf path, label). An uploaded PDF wins; else the NSE RHP; else say what is missing."""
    if pdf:
        return pdf, "uploaded document"
    dest = extract_rhp.RHP_DIR / f"{symbol}.pdf"
    if dest.exists():
        return dest, "RHP (cached)"
    url = find_rhp_url(detail)
    if not url:
        raise DataError("NSE has no RHP for this issue yet. Upload the RHP or DRHP PDF (SEBI: sebi.gov.in > Filings > "
                        "Public Issues) and try again.")
    extract_rhp.RHP_DIR.mkdir(parents=True, exist_ok=True)
    nse.download(url, dest)
    return dest, "RHP (NSE)"


def _date(v) -> pd.Timestamp:
    """NSE gives '23-Nov-2026'; our own files give '2026-03-27'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    return pd.to_datetime(v, errors="coerce", format="mixed", dayfirst=False)


def market_series() -> pd.DataFrame | None:
    """Stored Nifty/VIX closes, extended to today when Yahoo is reachable."""
    p = DATA / "market.csv"
    old = pd.read_csv(p, index_col="date", parse_dates=["date"]) if p.exists() else None
    try:
        new = index_closes(date.today() - timedelta(days=120), date.today())
    except Exception:
        new = None
    frames = [f for f in (old, new) if f is not None and len(f)]
    if not frames:
        return None
    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index)
    return out[~out.index.duplicated(keep="last")].sort_index()


def build_row(m: Match, detail, record: dict | None, hist: pd.DataFrame) -> pd.Series:
    """One-row frame in the same shape as data/ipos.csv, features derived the same way as training."""
    info = issue_info(detail) if detail else {}
    close = _date(m.issue_close)
    base = {"symbol": m.symbol, "company": m.company, "issue_open": _date(m.issue_open),
            "issue_close": close, "listing_date": close + pd.Timedelta(days=3) if pd.notna(close) else pd.NaT,
            "price_low": m.price_low, "price_high": m.price_high, "issue_price": m.price_high,
            "lot_size": info.get("lot_size_nse"), "lead_managers": info.get("lead_managers_nse"),
            "anchor_shares_cr": info.get("anchor_shares_cr")}
    fields = dict(record["fields"]) if record else {}
    fields = {k: v for k, v in fields.items() if not k.startswith("rhp_page_")}
    for k in ("company_rhp",):
        fields.pop(k, None)
    df = pd.DataFrame([{**fields, **{k: v for k, v in base.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}}])
    df = features.derive(df)
    mk = market_series()
    df = features.market_context(df, mk, history=hist[["listing_date", "listing_gain_pct"]])
    df["log_issue_size"] = np.log1p(df["issue_size_cr"])
    row = df.iloc[0]
    row = row.copy()
    row["rhp_checks"] = (record or {}).get("checks", [])
    return row


def analyse_issue(m: Match, pdf: Path | None = None, hist: pd.DataFrame | None = None,
                  nse: NSEClient | None = None, progress=lambda s: None) -> Analysis:
    nse = nse or NSEClient()
    hist = hist if hist is not None else load_history()
    notes: list[str] = []
    progress("Fetching NSE issue page")
    try:
        detail = nse.ipo_detail(m.symbol)
    except DataError as exc:
        detail = None
        notes.append(f"NSE issue page unavailable ({exc}).")
    record, source = None, "no document"
    try:
        progress("Getting the offer document")
        doc, source = get_document(m.symbol, detail, pdf, nse)
        progress("Reading it with Claude (a few minutes)")
        record = extract_rhp.extract(m.symbol, pdf=doc, client=extract_rhp.make_client())
    except (DataError, extract_rhp.ExtractionError, extract_rhp.RateLimited) as exc:
        notes.append(f"No fundamentals: {exc}")
    progress("Scoring")
    row = build_row(m, detail, record, hist)
    live = parse_bids(detail) if detail else {}
    live = {k: v for k, v in live.items() if pd.notna(v)}
    models = fit_prior_models(hist)
    X = pd.DataFrame([row]).reindex(columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce")
    n_have = int(X.notna().sum(axis=1).iloc[0])
    gain_p = sub_med = sub_rng = None
    if record and n_have >= 6:
        gain_p = float(models["clf"].predict_proba(X)[0, 1])
        mu = float(models["reg"].predict(X)[0])
        sd = models["resid_sd"]
        sub_med, sub_rng = np.expm1(mu), (np.expm1(mu - 1.28 * sd), np.expm1(mu + 1.28 * sd))
    elif record:
        notes.append("Too few fundamentals were extracted for a model estimate.")
    table = subscription_table(hist)
    price = row.get("issue_price")
    sc = table.copy()
    for c in ("p25", "median", "p75"):
        sc[f"price_{c}"] = price * (1 + sc[c] / 100) if pd.notna(price) else np.nan
    return Analysis(m, row, flags(row, hist), live, gain_p, sub_med, sub_rng, sc,
                    {k: models[k] for k in ("auc", "r2", "n_train", "n_test")}, table, source, notes)
