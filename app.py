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

from src import analyse, demo, live, schema

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


def page_new() -> None:
    st.header("Analyse a new IPO")
    st.caption("Type a company name or NSE symbol. The app finds it on NSE, reads the RHP with Claude, flags good and bad "
               "points and shows how similar past IPOs listed. Historical patterns only; not investment advice.")
    hp = ROOT / "data" / "ipos.csv"
    if not hp.exists():
        st.info("data/ipos.csv is missing; see README.md to build it.")
        return
    hist = hist_frame(hp.stat().st_mtime)
    c1, c2 = st.columns([3, 2])
    q = c1.text_input("IPO name or symbol", placeholder="e.g. Mahanadi Coalfields")
    up = c2.file_uploader("RHP / DRHP PDF (optional)", type="pdf", help="Use this when NSE has no RHP yet, or for a DRHP.")
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
    if st.button("Analyse"):
        pdf = None
        if up is not None:
            import tempfile
            pdf = Path(tempfile.mkdtemp()) / f"{m.symbol}.pdf"
            pdf.write_bytes(up.getvalue())
        with st.status("Working…", expanded=True) as status:
            try:
                st.session_state["analysis"] = live.analyse_issue(m, pdf=pdf, hist=hist, progress=status.write)
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(str(exc))
    a = st.session_state.get("analysis")
    if a is None or a.match.symbol != m.symbol:
        return
    r = a.row
    st.subheader(f"{a.match.company}")
    k = st.columns(4)
    k[0].metric("Price band", f"₹{a.match.price_low:g} – ₹{a.match.price_high:g}" if a.match.price_low else "–")
    k[1].metric("Issue size", f"₹{r['issue_size_cr']:,.0f} cr" if pd.notna(r.get("issue_size_cr")) else "–")
    k[2].metric("Offer for sale", f"{r['ofs_pct']:.0f}%" if pd.notna(r.get("ofs_pct")) else "–")
    k[3].metric("P/E at top of band", f"{r['pe_at_issue']:.1f}" if pd.notna(r.get("pe_at_issue")) else "–")
    for n in a.notes:
        st.warning(n)
    st.caption(f"Source: {a.source}.")

    if a.live_sub.get("sub_total_x") is not None:
        st.markdown("**Live subscription (NSE + BSE)**")
        cols = st.columns(5)
        for col, (key, label) in zip(cols, [("sub_total_x", "Total"), ("sub_qib_x", "QIB"), ("sub_nii_x", "NII"),
                                           ("sub_retail_x", "Retail"), ("sub_employee_x", "Employee")]):
            v = a.live_sub.get(key)
            col.metric(label, "–" if v is None else f"{v:.2f}x")
        st.caption("Institutions usually bid on the last day, so numbers are low until bidding closes. "
                   "Match the final number against the table below.")

    st.markdown("### Before bidding closes: rough prior from fundamentals")
    if a.prior_gain_prob is not None:
        p1, p2 = st.columns(2)
        p1.metric("Chance of listing above issue price", f"{a.prior_gain_prob:.0%}")
        lo, hi = a.prior_sub_range
        p2.metric("Likely final subscription", f"~{a.prior_sub_median:.0f}x", f"80% range {lo:.0f}x – {hi:.0f}x", delta_color="off")
        st.caption(live.PRE_BID_NOTE + f" Measured on this data: AUC {a.model_quality['auc']:.2f}, "
                   f"subscription R² {a.model_quality['r2']:.2f} on {a.model_quality['n_test']} IPOs from 2026.")
    else:
        st.info("No model estimate without extracted fundamentals.")

    st.markdown("### What past IPOs did, by final subscription")
    t = a.scenarios.copy()
    t.index.name = "final subscription"
    show = pd.DataFrame({"IPOs": t["n"], "Listed above issue": t["gain_rate"],
                         "Gain, weak 25%": t["p25"], "Gain, median": t["median"], "Gain, strong 25%": t["p75"],
                         "Price, weak": t["price_p25"], "Price, median": t["price_median"], "Price, strong": t["price_p75"]})
    st.dataframe(show.style.format({"Listed above issue": "{:.0%}", "Gain, weak 25%": "{:+.1f}%", "Gain, median": "{:+.1f}%",
                                    "Gain, strong 25%": "{:+.1f}%", "Price, weak": "₹{:,.0f}", "Price, median": "₹{:,.0f}",
                                    "Price, strong": "₹{:,.0f}"}), width="stretch")
    st.caption("Gain = listing-day open vs issue price. Price columns apply that gain to the top of the band. "
               "Allotment odds, costs and tax are not included.")

    good = [f for f in a.flags if f.kind == "good"]
    bad = [f for f in a.flags if f.kind == "bad"]
    info = [f for f in a.flags if f.kind == "info"]
    g, b = st.columns(2)
    with g:
        st.markdown("### 👍 Good")
        for f in good or []:
            st.markdown(f"**{f.text}**" + (f"<br><small>{f.evidence}</small>" if f.evidence else ""), unsafe_allow_html=True)
        if not good:
            st.write("Nothing stood out.")
    with b:
        st.markdown("### 👎 Bad")
        for f in bad or []:
            st.markdown(f"**{f.text}**" + (f"<br><small>{f.evidence}</small>" if f.evidence else ""), unsafe_allow_html=True)
        if not bad:
            st.write("Nothing stood out.")
    if info:
        st.markdown("### Worth knowing")
        for f in info:
            st.markdown(f"- {f.text}" + (f" ({f.evidence})" if f.evidence else ""))
    with st.expander("All extracted fields"):
        st.dataframe(r.reindex([c for c in schema.ORDER if c in r.index and schema.BY_NAME[c].group != "G"]).astype(str).to_frame("value"),
                     width="stretch", height=450)
    st.caption("Educational tool. Not investment advice. Past patterns do not guarantee future listings.")


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
