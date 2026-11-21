"""IPO listing-day analyser.

Run from this folder:  streamlit run app.py
Reads data/ipos.csv (built with `python -m src.build ...`). Educational tool, not investment advice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analyse, demo, extract_rhp, live, qualitative, report, schema

ROOT = Path(__file__).parent
st.set_page_config(page_title="IPO Listing-Day Analyser", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float) -> pd.DataFrame:
    return analyse.load(path)


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return analyse.add_split(demo.synthetic())


@st.cache_resource(show_spinner="Fitting rules and model on training rows…")
def run(df_key: str, _df: pd.DataFrame, target: str, quality: tuple[str, ...]) -> dict:
    return analyse.evaluate_all(_df, target=target, quality=quality)


# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📈 IPO listing-day")
    page = st.radio("View", ["Analyse a new IPO", "Dataset", "Winners vs losers", "Strategy test", "Check an IPO"])
    st.divider()
    seg = st.radio("Segment", ["Mainboard", "SME"], horizontal=True)
    path = ROOT / "data" / ("ipos.csv" if seg == "Mainboard" else "ipos_sme.csv")
    use_demo = st.toggle("Synthetic demo data", value=not path.exists(),
                         help="Fake companies with invented relationships, for trying the screens.")
    target = st.selectbox("Target", analyse.TARGETS,
                          format_func={"target_gain": "Listed above issue price", "target_gain_5": "Listed > 5% above"}.get)
    quality = tuple(st.multiselect("Rows to use", ["full", "partial", "missing_rhp"], default=["full", "partial"]))
    st.caption("Train = listed 2022–2025. Test = listed 2026 onward; it is never used to fit or choose anything.")


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def hist_frame(mtime: float) -> pd.DataFrame:
    return live.load_history()


@st.cache_resource(show_spinner="Fitting models on 2022-2025 and scoring 2026…")
def report_models(mtime: float):
    d = report.prepare(pd.read_csv(ROOT / "data" / "ipos.csv"))
    m = report.fit_models(d)
    return d, m, report.backtest(m, d)


def _pct(v, signed=False):
    return "–" if v is None or pd.isna(v) else (f"{v:+.0f}%" if signed else f"{v:.0%}")


def page_new() -> None:
    st.header("Analyse a new IPO")
    st.caption("Type a company name or NSE symbol. The app finds it on NSE, downloads its RHP (or use your own PDF), reads it with "
               "Claude, and reports the chance of a listing gain and the expected gain. Historical patterns only; not investment advice.")
    hp = ROOT / "data" / "ipos.csv"
    if not hp.exists():
        st.info("data/ipos.csv is missing; see README.md to build it.")
        return
    d, models, bt = report_models(hp.stat().st_mtime)
    hist = hist_frame(hp.stat().st_mtime)

    q = st.text_input("IPO name or symbol", placeholder="e.g. Mahanadi Coalfields")
    if st.button("Find", type="primary") and q.strip():
        try:
            st.session_state["matches"] = live.find_issue(q)
            st.session_state.pop("analysis", None)
        except Exception as exc:
            st.error(f"Could not reach NSE: {exc}")
    matches = st.session_state.get("matches", [])
    if not matches:
        return
    m = st.radio("Which one?", matches, format_func=lambda x: f"{x.company} ({x.symbol}) · {x.status} · {x.segment}")
    if m.segment == "SME":
        st.warning("This is an SME issue. The history here is mainboard only, so the numbers are not reliable for it.")

    c1, c2 = st.columns(2)
    up = c1.file_uploader("Upload the RHP (PDF), optional", type="pdf",
                          help="Use it when NSE is blocked or has no RHP link. It is saved as rhp/<SYMBOL>.pdf and replaces cached extractions if it differs.")
    with c2:
        st.markdown("**Subscription (overrides NSE's live figures)**")
        s1, s2 = st.columns(2)
        man = {"sub_qib_x": s1.number_input("QIB (x)", min_value=0.0, value=0.0, step=0.1),
               "sub_nii_x": s2.number_input("NII (x)", min_value=0.0, value=0.0, step=0.1),
               "sub_retail_x": s1.number_input("Retail (x)", min_value=0.0, value=0.0, step=0.1),
               "sub_total_x": s2.number_input("Total (x)", min_value=0.0, value=0.0, step=0.1)}
        closed = st.checkbox("Bidding has closed (these are final figures)")
    with_qual = st.checkbox("Include business, risks and litigation (a second Claude pass, uses more of your plan)", value=True)

    if st.button("Analyse"):
        pdf = None
        if up is not None:
            import tempfile
            pdf = Path(tempfile.mkdtemp()) / f"{m.symbol}.pdf"
            pdf.write_bytes(up.getvalue())
        with st.status("Working…", expanded=True) as status:
            try:
                a = live.analyse_issue(m, pdf=pdf, hist=hist, progress=status.write)
                qual = None
                if with_qual and a.record is not None:
                    status.write("Reading business, risks and litigation")
                    try:
                        qual = qualitative.extract_qual(m.symbol, pdf=extract_rhp.RHP_DIR / f"{m.symbol}.pdf", client=extract_rhp.make_client())
                    except Exception as exc:
                        a.notes.append(f"Business and risk pass failed: {exc}")
                st.session_state["analysis"], st.session_state["qual"] = a, qual
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(str(exc))
    a = st.session_state.get("analysis")
    if a is None or a.match.symbol != m.symbol:
        return
    sub = {**a.live_sub, **{k: v for k, v in man.items() if v > 0}}
    rep = report.build_report(a.row, a.record, st.session_state.get("qual"), models, d, sub, closed, backtest_res=bt)
    rep.notes += a.notes
    e, gq = rep.est, rep.est["gain"]
    st.title(rep.company)
    st.caption(f"Source: {a.source}. Stage: {e['stage']}.")
    for n in rep.notes:
        st.warning(n)

    st.subheader("1. Verdict")
    {"Apply": st.success, "Apply with caution": st.warning, "Skip": st.error}.get(rep.verdict, st.info)(f"**{rep.verdict}**")
    for x in rep.reasons:
        st.markdown(f"- {x}")

    st.subheader("2. Listing chance and expected gain")
    k = st.columns(3)
    k[0].metric(f"Chance of a listing gain ({e['stage']})", _pct(e["prob"]) + (" or more" if e["floor"] else ""))
    k[1].metric("Expected listing gain (median)", _pct(gq["p50"], True), f"likely {_pct(gq['p25'], True)} to {_pct(gq['p75'], True)}", delta_color="off")
    k[2].metric("Implied listing price", f"₹{e['prices']['p50']:,.0f}" if pd.notna(e["prices"]["p50"]) else "–",
                f"₹{e['prices']['p25']:,.0f} – ₹{e['prices']['p75']:,.0f}" if pd.notna(e["prices"]["p25"]) else "", delta_color="off")
    k = st.columns(3)
    k[0].metric("80% range for the gain", f"{_pct(gq['p10'], True)} to {_pct(gq['p90'], True)}")
    k[1].metric("Allotment odds per application", _pct(e["p_allot"]))
    k[2].metric("Expected profit per lot", "–" if pd.isna(e["profit_per_lot"]) else f"₹{e['profit_per_lot']:,.0f}", f"lot ₹{e['lot_value']:,.0f}", delta_color="off")
    if e["stage"] == "pre-bid":
        st.info("Pre-bid estimate from fundamentals and the 10 most similar past IPOs. This is a weak signal; the range is wide on purpose.")
    elif e["floor"]:
        st.info("Provisional: live figures are a floor because QIBs bid mostly on the last day. Tick 'bidding has closed' when final.")
    need = e["qib_needed"]
    st.caption("QIB subscription needed for a 70% / 85% / 95% chance: " + " / ".join("over 3000x" if need[t] is None else f"{need[t]:.0f}x" for t in (0.70, 0.85, 0.95))
               + f".  Gain model used: {e['gain_model']}.")
    if bt.get("demand"):
        st.caption(f"Scored on {bt['n_test']} IPOs from 2026 (models never saw them): subscription model AUC {bt['demand']['auc']:.2f}, accuracy {bt['demand']['accuracy']:.0%}; "
                   f"fundamentals-only AUC {bt['prebid']['auc']:.2f}; gain median error {bt['gain_a']['mae_p50']:.1f} points; "
                   f"25-75 range covers {bt['gain_a']['cov_25_75']:.0%}, 10-90 covers {bt['gain_a']['cov_10_90']:.0%}. Full backtest below.")

    st.subheader("3. Subscription")
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("sub_total_x", "Total"), ("sub_qib_x", "QIB"), ("sub_nii_x", "NII"), ("sub_retail_x", "Retail")]):
        col.metric(label, "–" if sub.get(key) is None else f"{sub[key]:.2f}x")
    st.dataframe(rep.band_table.style.format({"listed above issue": "{:.0%}", "median gain %": "{:+.1f}", "p25 %": "{:+.1f}", "p75 %": "{:+.1f}"}), width="stretch")

    st.subheader("4. The business")
    qq = rep.qual
    pt = lambda p: p["text"] + (f" *(PDF p.{p['page']})*" if p.get("page") else "")
    if qq:
        for key, title in (("business_summary", "What it does"), ("industry", "Industry")):
            st.markdown(f"**{title}**")
            for p in qq.get(key, []):
                st.markdown(f"- {pt(p)}")
        if qq.get("segments"):
            st.markdown("**Segments**")
            for s in qq["segments"]:
                st.markdown(f"- {s['name']}" + (f": {s['share_of_revenue']}" if s.get("share_of_revenue") else "") + (f" *(p.{s['page']})*" if s.get("page") else ""))
        st.markdown("**Concentration**")
        for p in qq.get("concentration", []):
            st.markdown(f"- {pt(p)}")
        st.markdown("**Promoters**")
        for p in qq.get("promoters", []):
            st.markdown(f"- {p['name']}: {p['background']}" + (f" *(p.{p['page']})*" if p.get("page") else ""))
    else:
        st.info("Business, risk and litigation pass not run (tick the box above and press Analyse).")

    st.subheader("5. Key numbers")
    st.table(pd.DataFrame(rep.key_numbers, columns=["Item", "Value", "Note"]).set_index("Item"))
    st.markdown("**Financials (₹ crore)**")
    st.dataframe(rep.financials.style.format("{:,.1f}", na_rep="–"), width="stretch")
    st.markdown("**Listed peers**")
    st.dataframe(rep.peers, hide_index=True, width="stretch")
    p1, p2 = st.columns(2)
    p1.markdown("**Use of fresh money**")
    p1.dataframe(rep.proceeds, hide_index=True, width="stretch")
    p2.markdown("**Selling shareholders**")
    p2.dataframe(rep.sellers, hide_index=True, width="stretch")

    st.subheader("6. Scorecard")
    st.dataframe(rep.scorecard.style.format({"score_0_10": "{:.1f}", "train_spearman_vs_gain": "{:+.2f}", "train_auc": "{:.2f}"}, na_rep="–"), width="stretch")
    st.caption("Score = percentile rank against every past IPO, scaled 0-10. The last two columns show how well that pillar predicted listing results on 2022-2025 (AUC 0.5 = no skill).")

    st.subheader("7. Strengths, concerns and risks")
    gcol, bcol = st.columns(2)
    with gcol:
        st.markdown("**Strengths**")
        for f in [f for f in rep.flags if f.kind == "good"]:
            st.markdown(f"- {f.text}" + (f" *({f.evidence})*" if f.evidence else ""))
        for p in (qq or {}).get("strengths", []):
            st.markdown(f"- {pt(p)}")
    with bcol:
        st.markdown("**Concerns**")
        for f in [f for f in rep.flags if f.kind == "bad"]:
            st.markdown(f"- {f.text}" + (f" *({f.evidence})*" if f.evidence else ""))
    if qq:
        st.markdown("**Top company-specific risks**")
        for x in qq.get("risks", []):
            st.markdown(f"- **[{x['severity']}]** {x['title']}: {x['detail']}" + (f" *(p.{x['page']})*" if x.get("page") else ""))
        for key, title in (("litigation", "Litigation"), ("contingent_liabilities", "Contingent liabilities"), ("related_party", "Related-party transactions"), ("red_flags", "Red flags")):
            items = qq.get(key, [])
            st.markdown(f"**{title}**")
            if not items:
                st.write("None stated in the pages read.")
            for x in items:
                if key == "litigation":
                    st.markdown(f"- {x['category']}: {x.get('count') or '?'} cases, {x.get('amount') or 'amount not stated'}" + (f" *(p.{x['page']})*" if x.get("page") else ""))
                else:
                    st.markdown(f"- {pt(x)}")

    st.subheader("8. Most similar past IPOs")
    st.dataframe(rep.similar.assign(listing_date=rep.similar["listing_date"].dt.date), hide_index=True, width="stretch",
                 column_config={"distance": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download report (Markdown)", report.to_markdown(rep), file_name=f"{rep.symbol}_report.md", mime="text/markdown")

    with st.expander("Backtest: 2022-2025 train, 2026 test", expanded=False):
        show_backtest(bt)
    st.caption("Educational tool. Not investment advice. Past patterns do not guarantee future listings.")


def show_backtest(bt: dict) -> None:
    if not bt.get("demand"):
        st.info("Not enough 2026 rows to score.")
        return
    st.markdown(f"**{bt['n_test']} IPOs listed in 2026. Models were fitted on {bt['n_train']} IPOs from 2022-2025 and never saw these.**")
    rows = {"Subscription model (QIB, total, retail, PAT margin)": bt["demand"], "Fundamentals only (pre-bid)": bt["prebid"]}
    st.dataframe(pd.DataFrame(rows).T[["n", "auc", "brier", "accuracy", "base_rate"]].style.format({"auc": "{:.2f}", "brier": "{:.3f}", "accuracy": "{:.0%}", "base_rate": "{:.0%}"}), width="stretch")
    st.dataframe(bt["baselines"].style.format("{:.0%}"), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("Calibration, subscription model (deciles)")
    c1.dataframe(bt["calibration_demand"].style.format({"predicted": "{:.0%}", "actual": "{:.0%}"}), width="stretch")
    c2.markdown("Calibration, pre-bid model (deciles)")
    c2.dataframe(bt["calibration_prebid"].style.format({"predicted": "{:.0%}", "actual": "{:.0%}"}), width="stretch")
    g = pd.DataFrame({"Model A (quantile regression)": bt["gain_a"], "Model B (similar QIB)": bt["gain_b"], "Pre-bid (10 nearest IPOs)": bt["gain_prebid_knn"]}).T
    st.markdown(f"Listing-gain %: Model {bt['gain_model_default']} is the default (chosen on a 2025 validation fold inside the training data).")
    st.dataframe(g.style.format({"mae_p50": "{:.1f}", "cov_25_75": "{:.0%}", "cov_10_90": "{:.0%}", "spearman": "{:.2f}", "pinball": "{:.2f}"}), width="stretch")
    st.caption("mae_p50 = median absolute error of the 50th-percentile prediction, in percentage points. The 25-75 and 10-90 ranges should cover about 50% and 80%.")
    st.markdown("Strategies on 2026")
    st.dataframe(bt["strategies"].style.format({"hit_rate": "{:.0%}", "avg_gain_pct": "{:+.1f}", "allotment_weighted_gain_pct": "{:+.1f}", "sum_expected_gain_per_lot_rs": "₹{:,.0f}"}), width="stretch")


if page == "Analyse a new IPO":
    page_new()
    st.stop()

if use_demo:
    df, key = load_demo(), "demo"
    st.warning("Synthetic demo data: every company and number is fake.", icon="⚠️")
elif path.exists():
    df, key = load(str(path), path.stat().st_mtime), f"{path}:{path.stat().st_mtime}"
else:
    st.info(f"No {path.relative_to(ROOT)} yet. Build it with the steps in README.md, or turn on the demo data.")
    st.stop()

pct = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


# --------------------------------------------------------------------------- #
def page_dataset() -> None:
    st.header("Dataset")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    k = st.columns(5)
    k[0].metric("IPOs", len(df))
    k[1].metric("Train / test", f"{len(train)} / {len(test)}")
    k[2].metric("Full quality", int((df["data_quality"] == "full").sum()))
    br = analyse.base_rate(train, target)
    k[3].metric("Base rate (train)", pct(br["rate"]), f"{br['positive']} of {br['n']}", delta_color="off")
    k[4].metric("Median listing gain (train)", f"{train['listing_gain_pct'].median():+.1f}%")
    st.caption("Base rate = share of training IPOs that hit the target. Any strategy has to beat 'always apply', which scores exactly this.")

    by_year = (df.assign(year=pd.to_datetime(df["listing_date"]).dt.year)
               .groupby("year").agg(ipos=("symbol", "size"), hit_rate=(target, "mean"),
                                    median_gain=("listing_gain_pct", "median"),
                                    full=("data_quality", lambda s: (s == "full").sum())))
    st.dataframe(by_year.style.format({"hit_rate": "{:.0%}", "median_gain": "{:+.1f}%"}), width="stretch")

    fig = px.histogram(df, x="listing_gain_pct", color="split", nbins=50, barmode="overlay",
                       labels={"listing_gain_pct": "Listing gain % (open vs issue price)"})
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Coverage by column"):
        cov = df.reindex(columns=schema.ORDER).notna().mean().rename("filled").to_frame()
        cov["group"] = [schema.BY_NAME[c].group for c in cov.index]
        cov["source"] = [schema.BY_NAME[c].source for c in cov.index]
        st.dataframe(cov.style.format({"filled": "{:.0%}"}), width="stretch", height=400)
    with st.expander("All rows"):
        st.dataframe(df, hide_index=True, width="stretch")


def page_groups() -> None:
    st.header("Winners vs losers (training rows only)")
    res = run(key, df, target, quality)
    g = res["groups"]
    if g.empty:
        st.info("Not enough training rows with outcomes.")
        return
    st.caption("For each feature: median among IPOs that hit the target vs those that didn't, and the share of each "
               "group above the overall median. A big share gap means the feature separates the groups. "
               "p-value is Mann–Whitney; with ~80 features, expect a few p < 0.05 by chance.")
    st.dataframe(g, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in g.columns if c != "feature"})
    f = st.selectbox("Look at one feature", g["feature"])
    d = df[(df["split"] == "train")].assign(outcome=lambda x: x[target].map({1: "hit", 0: "miss"}))
    c1, c2 = st.columns(2)
    fig = px.box(d, x="outcome", y=f, points="all", hover_name="company", color="outcome")
    fig.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    c1.plotly_chart(fig, width="stretch")
    fig = px.scatter(d, x=f, y="listing_gain_pct", hover_name="company", color="outcome",
                     log_x=bool((pd.to_numeric(d[f], errors="coerce") > 0).all() and f.startswith("sub_")))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    c2.plotly_chart(fig, width="stretch")


def page_strategy() -> None:
    st.header("Strategy test")
    res = run(key, df, target, quality)
    if "error" in res:
        st.warning(res["error"])
        return
    st.markdown(f"**Rule set chosen on 2022–2025:** `{res['rules']}`")
    st.caption("Greedy search: add one threshold at a time while training precision improves and the rule still "
               "selects at least 25% of training IPOs.")
    fmt = {"accuracy": "{:.0%}", "precision": "{:.0%}", "recall": "{:.0%}",
           "avg_gain_if_applied_pct": "{:+.1f}%", "avg_gain_all_pct": "{:+.1f}%"}
    for title, tab in (("2026 test (out of sample)", res["test_scores"]), ("Training (in sample, optimistic)", res["train_scores"])):
        st.subheader(title)
        if tab.empty:
            st.info("No test rows yet.")
            continue
        st.dataframe(tab.astype(float).style.format(fmt | {c: "{:.0f}" for c in ("n", "tp", "fp", "tn", "fn", "n_apply")}),
                     width="stretch")
    st.caption("precision = of the IPOs a strategy says to apply for, the share that hit the target. "
               "avg_gain_if_applied = mean listing gain of those IPOs (before allotment odds, costs and tax).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Best single thresholds")
        st.dataframe(res["single_conditions"].assign(condition=lambda x: x["condition"].astype(str)),
                     hide_index=True, width="stretch")
    with c2:
        st.subheader("Logistic regression weights")
        st.dataframe(res["coefficients"].head(20), hide_index=True, width="stretch")
        st.caption("Standardised, so sizes are comparable. 'missingindicator_*' = the value was missing.")


def page_check() -> None:
    st.header("Check an IPO against the strategy")
    res = run(key, df, target, quality)
    if "error" in res:
        st.warning(res["error"])
        return
    options = df.sort_values("listing_date", ascending=False)
    pick = st.selectbox("IPO", options.index, format_func=lambda i: f"{options.at[i, 'company']} ({options.at[i, 'listing_date']:%d %b %Y}, {options.at[i, 'split']})")
    row = df.loc[[pick]]
    prob = float(analyse.predict_proba(res["model"], row)[0])
    passes = [(str(c), bool(c.mask(row)[0])) for c in res["rules"].conditions]
    k = st.columns(4)
    k[0].metric("Model probability of hit", f"{prob:.0%}")
    k[1].metric("Rule set", "Apply" if all(p for _, p in passes) else "Skip")
    k[2].metric("Actual listing gain", "–" if pd.isna(row["listing_gain_pct"].iloc[0]) else f"{row['listing_gain_pct'].iloc[0]:+.1f}%")
    k[3].metric("Data quality", row["data_quality"].iloc[0])
    for cond, ok in passes:
        st.markdown(f"{'✅' if ok else '❌'} `{cond}`")
    if row["split"].iloc[0] == "train":
        st.caption("This IPO was in the training data, so the model has already seen its outcome.")
    show = row.reindex(columns=[c for c in schema.ORDER if schema.BY_NAME[c].group != "G"]).T.rename(columns={pick: "value"})
    st.dataframe(show.astype(str), width="stretch", height=500)


{"Dataset": page_dataset, "Winners vs losers": page_groups,
 "Strategy test": page_strategy, "Check an IPO": page_check}[page]()
