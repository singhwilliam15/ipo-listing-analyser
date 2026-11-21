"""IPO listing-day analyser.

Run from this folder:  streamlit run app.py
Reads data/ipos.csv (built with `python -m src.build ...`). Educational tool, not investment advice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analyse, demo, extract_rhp, live, qualitative, report, schema

ROOT = Path(__file__).parent
st.set_page_config(page_title="IPO Listing-Day Analyser", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


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
    st.title("📈 IPO Listing-Day Analyser")
    st.caption("Indian mainboard IPOs: chance of a listing gain, expected gain, and why.")
    page = st.radio("Go to", ["Analyse a new IPO", "Model accuracy", "Past IPOs", "Winners vs losers", "Strategy test", "Check a past IPO"], label_visibility="collapsed")
    research = page not in ("Analyse a new IPO", "Model accuracy")
    seg, target, quality, use_demo = "Mainboard", "target_gain", ("full", "partial"), False
    if research:
        st.divider()
        seg = st.radio("Segment", ["Mainboard", "SME"], horizontal=True)
        target = st.selectbox("Target", analyse.TARGETS,
                              format_func={"target_gain": "Listed above issue price", "target_gain_5": "Listed > 5% above"}.get)
        quality = tuple(st.multiselect("Rows to use", ["full", "partial", "missing_rhp"], default=["full", "partial"]))
    path = ROOT / "data" / ("ipos.csv" if seg == "Mainboard" else "ipos_sme.csv")
    if research:
        use_demo = st.toggle("Synthetic demo data", value=not path.exists(), help="Fake companies, only for trying the screens.")
    st.divider()
    st.caption("Live estimates use models refit on all IPOs to date. Accuracy is measured by a walk-forward: each year is scored by models trained only on earlier years. Educational tool, not investment advice.")


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def hist_frame(mtime: float) -> pd.DataFrame:
    return live.load_history()


@st.cache_resource(show_spinner="Fitting models and running the walk-forward backtest…")
def report_models(mtime: float):
    """published = trained on 2022-2025 (a single 2026 test); production = refit on every IPO to date and used for live estimates. Accuracy comes from the walk-forward."""
    d = report.prepare(pd.read_csv(ROOT / "data" / "ipos.csv"))
    wf_raw = report.walk_forward_raw(d)
    published = report.fit_models(d, wf_raw=wf_raw)
    production = report.fit_models(d, cutoff=None, wf_raw=wf_raw)
    return d, published, production, report.backtest(published, d), report.walk_forward(d, wf_raw)


def _pct(v, signed=False):
    return "–" if v is None or pd.isna(v) else (f"{v:+.0f}%" if signed else f"{v:.0%}")


@st.cache_data(ttl=600, show_spinner=False)
def open_issues() -> pd.DataFrame:
    return live.list_open_issues()


def page_new() -> None:
    st.markdown("# Will this IPO list at a gain?")
    st.markdown("Type an Indian mainboard IPO. The app finds it on NSE, downloads its RHP, reads it with Claude, and estimates the "
                "**chance of a listing gain** and the **expected gain in %**, with the reasons, risks and the most similar past IPOs. "
                "It learns from 390+ IPOs listed 2022-2026.")
    hp = ROOT / "data" / "ipos.csv"
    try:
        opened = open_issues()
    except Exception:
        opened = pd.DataFrame()
    if len(opened):
        mb = opened[opened["segment"] == "Mainboard"]
        if len(mb):
            with st.expander(f"Open or upcoming mainboard IPOs on NSE ({len(mb)})", expanded=False):
                st.dataframe(mb[["company", "symbol", "status", "price_low", "price_high", "issue_open", "issue_close"]], hide_index=True, width="stretch")
    if not hp.exists():
        st.info("data/ipos.csv is missing; see README.md to build it.")
        return
    d, _published, models, bt, wf = report_models(hp.stat().st_mtime)
    hist = hist_frame(hp.stat().st_mtime)

    with st.expander("Search and inputs", expanded=st.session_state.get("analysis") is None):
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

        ran = False
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
                    ran = True
                except Exception as exc:
                    status.update(label="Failed", state="error")
                    st.error(str(exc))
        if ran:
            st.rerun()                      # redraw with the search form folded away, report on top
    a = st.session_state.get("analysis")
    if a is None or a.match.symbol != m.symbol:
        return
    sub = {**a.live_sub, **{k: v for k, v in man.items() if v > 0}}
    rep = report.build_report(a.row, a.record, st.session_state.get("qual"), models, d, sub, closed, backtest_res={**bt, "walk_forward": wf})
    rep.notes += a.notes
    render_report(rep, a, sub, bt, wf, models.info["n_train"])


VERDICT_STYLE = {"Apply": ("#137333", "#e6f4ea"), "Apply with caution": ("#b06000", "#fef7e0"),
                 "Skip": ("#c5221f", "#fce8e6"), "Wait for subscription": ("#1a73e8", "#e8f0fe")}


def gain_range_chart(e: dict) -> go.Figure:
    g = e["gain"]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=g["p10"], x1=g["p90"], y0=-0.25, y1=0.25, fillcolor="rgba(26,115,232,0.22)", line_width=0)
    fig.add_shape(type="rect", x0=g["p25"], x1=g["p75"], y0=-0.25, y1=0.25, fillcolor="rgba(26,115,232,0.60)", line_width=0)
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.add_trace(go.Scatter(x=[g["p50"]], y=[0], mode="markers+text", text=[f"{g['p50']:+.0f}%"], textposition="top center",
                             marker=dict(size=14, color="#0b3d91", symbol="diamond"), hoverinfo="skip", showlegend=False))
    fig.update_yaxes(visible=False, range=[-0.6, 0.8])
    fig.update_xaxes(title="Listing gain vs issue price (%)  ·  light = 80% range, dark = likely range", zeroline=False)
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=6, b=0))
    return fig


def band_chart(rep) -> go.Figure:
    t = rep.band_table.reset_index().rename(columns={"band": "final subscription"})
    now = rep.sub.get("sub_total_x")
    labels = list(t["final subscription"].astype(str))
    cur = None
    if now:
        cur = labels[np.searchsorted([3, 10, 30, 100], now)]
    colors = ["#1a73e8" if lab == cur else "#aecbfa" for lab in labels]
    fig = go.Figure(go.Bar(x=labels, y=t["listed above issue"], marker_color=colors, text=[f"{v:.0%}" for v in t["listed above issue"]], textposition="outside",
                           customdata=t[["IPOs", "median gain %"]], hovertemplate="%{x}<br>%{y:.0%} listed above issue<br>%{customdata[0]} IPOs, median %{customdata[1]:+.1f}%<extra></extra>"))
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title="Listed above issue price")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Final total subscription")
    return fig


def scorecard_chart(sc: pd.DataFrame) -> go.Figure:
    s = sc["score_0_10"]
    fig = go.Figure(go.Bar(y=list(sc.index)[::-1], x=s.fillna(0).iloc[::-1], orientation="h", marker_color=["#1a73e8" if pd.notna(v) else "#dadce0" for v in s.iloc[::-1]],
                           text=["n/a" if pd.isna(v) else f"{v:.1f}" for v in s.iloc[::-1]], textposition="outside"))
    fig.update_xaxes(range=[0, 11], title="Percentile rank vs all past IPOs, 0-10")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def render_report(rep, a, sub: dict, bt: dict, wf: dict, n_live_train: int) -> None:
    e, gq = rep.est, rep.est["gain"]
    head, dl = st.columns([5, 1])
    head.markdown(f"## {rep.company}")
    head.caption(f"NSE: {rep.symbol} · source: {a.source} · stage: **{e['stage']}**" + (" (live figures are a floor)" if e["floor"] else ""))
    dl.download_button("Download report", report.to_markdown(rep), file_name=f"{rep.symbol}_report.md", mime="text/markdown", use_container_width=True)
    for n in rep.notes:
        st.warning(n)

    fg, bgc = VERDICT_STYLE.get(rep.verdict, ("#202124", "#f1f3f4"))
    reasons = "".join(f"<li>{x}</li>" for x in rep.reasons)
    st.markdown(f"<div style='border-left:6px solid {fg};background:{bgc};padding:14px 18px;border-radius:8px;color:#202124'>"
                f"<div style='font-size:0.8rem;letter-spacing:.06em;color:{fg};font-weight:600'>VERDICT</div>"
                f"<div style='font-size:1.6rem;font-weight:700;color:{fg}'>{rep.verdict}</div><ul style='margin:6px 0 0 0'>{reasons}</ul></div>", unsafe_allow_html=True)
    st.write("")
    need = e["qib_needed"]
    need_txt = lambda t: "over 3000x" if need[t] is None else f"{need[t]:.0f}x"
    if e["stage"] == "provisional":
        st.markdown("#### QIB subscription needed by the close")
        n3 = st.columns(3)
        now = sub.get("sub_qib_x")
        for col, t in zip(n3, (0.70, 0.85, 0.95)):
            col.metric(f"For a {t:.0%} chance", need_txt(t), f"now {now:.1f}x" if now else "", delta_color="off")
        st.caption("Bidding is open and QIBs bid mostly on the last day, so the headline below stays the pre-bid estimate.")
    k = st.columns(4)
    k[0].metric(f"Chance of a listing gain ({e['headline']})", e["prob_text"],
                help="Probability the listing-day open is above the issue price. Before bidding, anything below 80% is shown as about the base rate, because the pre-bid model cannot separate IPOs that finely.")
    k[1].metric("Expected listing gain", _pct(gq["p50"], True), f"likely {_pct(gq['p25'], True)} to {_pct(gq['p75'], True)}", delta_color="off")
    k[2].metric("Implied listing price", f"₹{e['prices']['p50']:,.0f}" if pd.notna(e["prices"]["p50"]) else "–",
                f"₹{e['prices']['p25']:,.0f} – ₹{e['prices']['p75']:,.0f}" if pd.notna(e["prices"]["p25"]) else "", delta_color="off")
    if e["stage"] == "final":
        k[3].metric("Expected profit per lot", "–" if pd.isna(e["profit_per_lot"]) else f"₹{e['profit_per_lot']:,.0f}",
                    f"allotment odds {_pct(e['p_allot'])}", delta_color="off", help="Allotment odds × expected gain × lot value.")
    else:
        k[3].metric("Allotment odds, profit per lot", "after close", help="Shown once bidding has closed and the final figures are in.")
    st.plotly_chart(gain_range_chart(e), width="stretch", config={"displayModeBar": False})
    ic = e.get("if_closed")
    if ic:
        st.info(f"**If bidding closed at today's figures:** {_pct(ic['prob'])} chance of a listing gain, median gain {_pct(ic['gain']['p50'], True)} "
                f"(likely {_pct(ic['gain']['p25'], True)} to {_pct(ic['gain']['p75'], True)}). A floor, because subscription is still building. "
                "Tick 'bidding has closed' in the inputs when it is final.")
    elif e["stage"] == "pre-bid":
        st.info("Pre-bid estimate from fundamentals and the 10 most similar past IPOs. This is a weak signal, so the range is wide on purpose.")
    else:
        st.caption("QIB subscription needed for a 70% / 85% / 95% chance: " + " / ".join(need_txt(t) for t in (0.70, 0.85, 0.95)) + f" · gain model {e['gain_model']}")
    st.caption(f"Estimates for this IPO come from models refit on all {n_live_train} IPOs to date, which cannot be tested out of sample. "
               "Their accuracy is measured by the walk-forward: each of 2024, 2025 and 2026 scored by models trained only on earlier years (Model accuracy tab).")
    if wf and "pooled" in wf["prob"].index:
        pw, ga = wf["prob"].loc["pooled"], wf["gain"].loc[("A", "pooled")]
        st.caption(f"Walk-forward, pooled over {int(pw['n'])} IPOs listed 2024-2026: subscription model AUC {pw['demand_auc']:.2f}, accuracy {pw['demand_acc_raw']:.0%}; "
                   f"pre-bid AUC {pw['prebid_auc']:.2f}; gain median error {ga['mae_p50']:.1f} points. See the Model accuracy tab.")

    qq = rep.qual
    pt = lambda p: p["text"] + (f" *(PDF p.{p['page']})*" if p.get("page") else "")
    tabs = st.tabs(["Subscription", "Business", "Numbers", "Scorecard", "Strengths & risks", "Similar IPOs", "Model accuracy"])
    with tabs[0]:
        for row_cats in ([("sub_total_x", "Total"), ("sub_qib_x", "QIB"), ("sub_nii_x", "NII"), ("sub_retail_x", "Retail")],
                         [("sub_employee_x", "Employee quota"), ("sub_shareholder_x", "Shareholder quota")]):
            for col, (key, label) in zip(st.columns(4), row_cats):
                col.metric(label, "–" if sub.get(key) is None else f"{sub[key]:.2f}x")
        st.markdown("**How past IPOs listed at each final subscription level**")
        st.plotly_chart(band_chart(rep), width="stretch", config={"displayModeBar": False})
        st.dataframe(rep.band_table.style.format({"listed above issue": "{:.0%}", "median gain %": "{:+.1f}", "p25 %": "{:+.1f}", "p75 %": "{:+.1f}"}), width="stretch")
    with tabs[1]:
        if qq:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**What it does**")
                for p in qq.get("business_summary", []):
                    st.markdown(f"- {pt(p)}")
                st.markdown("**Promoters**")
                for p in qq.get("promoters", []):
                    st.markdown(f"- {p['name']}: {p['background']}" + (f" *(p.{p['page']})*" if p.get("page") else ""))
            with c2:
                st.markdown("**Industry**")
                for p in qq.get("industry", []):
                    st.markdown(f"- {pt(p)}")
                st.markdown("**Concentration**")
                for p in qq.get("concentration", []):
                    st.markdown(f"- {pt(p)}")
            if qq.get("segments"):
                st.markdown("**Segments**")
                for s in qq["segments"]:
                    st.markdown(f"- {s['name']}" + (f": {s['share_of_revenue']}" if s.get("share_of_revenue") else "") + (f" *(p.{s['page']})*" if s.get("page") else ""))
        else:
            st.info("The business, risk and litigation pass was not run. Tick the box above the Analyse button and press Analyse.")
    with tabs[2]:
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
    with tabs[3]:
        st.plotly_chart(scorecard_chart(rep.scorecard), width="stretch", config={"displayModeBar": False})
        st.dataframe(rep.scorecard.style.format({"score_0_10": "{:.1f}", "train_spearman_vs_gain": "{:+.2f}", "train_auc": "{:.2f}"}, na_rep="–"), width="stretch")
        st.caption("Score = percentile rank against every past IPO. The last two columns show how well each pillar predicted listing results on 2022-2025 (AUC 0.5 = no skill). Demand predicts best; the others add little on their own.")
    with tabs[4]:
        gcol, bcol = st.columns(2)
        with gcol:
            st.markdown("#### 👍 Strengths")
            for f in [f for f in rep.flags if f.kind == "good"]:
                st.markdown(f"- {f.text}" + (f" *({f.evidence})*" if f.evidence else ""))
            for p in (qq or {}).get("strengths", []):
                st.markdown(f"- {pt(p)}")
        with bcol:
            st.markdown("#### 👎 Concerns")
            for f in [f for f in rep.flags if f.kind == "bad"]:
                st.markdown(f"- {f.text}" + (f" *({f.evidence})*" if f.evidence else ""))
        if qq:
            st.markdown("#### Top company-specific risks")
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
            for x in qq.get("risks", []):
                st.markdown(f"- {icon.get(x['severity'], '')} **{x['title']}**: {x['detail']}" + (f" *(p.{x['page']})*" if x.get("page") else ""))
            for key, title in (("litigation", "Litigation"), ("contingent_liabilities", "Contingent liabilities"), ("related_party", "Related-party transactions"), ("red_flags", "Red flags")):
                with st.expander(title, expanded=key == "red_flags"):
                    items = qq.get(key, [])
                    if not items:
                        st.write("None stated in the pages read.")
                    for x in items:
                        if key == "litigation":
                            st.markdown(f"- {x['category']}: {x.get('count') or '?'} cases, {x.get('amount') or 'amount not stated'}" + (f" *(p.{x['page']})*" if x.get("page") else ""))
                        else:
                            st.markdown(f"- {pt(x)}")
    with tabs[5]:
        st.caption("The 10 past IPOs whose fundamentals (and subscription, once known) are closest to this one. Same sector counts as closer.")
        st.dataframe(rep.similar.assign(listing_date=rep.similar["listing_date"].dt.date), hide_index=True, width="stretch",
                     column_config={"distance": st.column_config.NumberColumn(format="%.2f"), "listing_gain_pct": st.column_config.NumberColumn("listing gain %", format="%+.1f")})
    with tabs[6]:
        show_backtest(bt, wf)
    st.caption("Educational tool, not investment advice. Models learn from past listings and can be wrong.")


def show_walk_forward(wf: dict) -> None:
    if not wf or not len(wf.get("years", [])):
        return
    st.markdown("### Walk-forward: every test year scored by models trained on all earlier years")
    st.caption("2024 is tested by models trained on 2022-2023, 2025 on 2022-2024, 2026 on 2022-2025. 'pooled' scores all test years together. "
               "Live estimates use models refit on every IPO to date, so this is the evidence for them.")
    pr = wf["prob"]
    st.markdown("**Once subscription is final** (subscription model)")
    t = pr[["n", "base_rate", "demand_auc", "demand_brier_raw", "demand_acc_raw"]].rename(columns={
        "n": "IPOs", "base_rate": "base rate", "demand_auc": "AUC", "demand_brier_raw": "Brier", "demand_acc_raw": "accuracy"})
    st.dataframe(t.style.format({"IPOs": "{:.0f}", "base rate": "{:.0%}", "AUC": "{:.2f}", "Brier": "{:.3f}", "accuracy": "{:.0%}"}), width="stretch")
    st.markdown("**Before bidding opens**: new method (forecast final QIB, push 21 points through the models) vs the previous one (logistic on fundamentals)")
    t = pr[["prebid_auc", "prebid_brier", "prebid_acc", "old_prebid_auc", "old_prebid_brier", "old_prebid_acc"]].rename(columns={
        "prebid_auc": "new AUC", "prebid_brier": "new Brier", "prebid_acc": "new accuracy",
        "old_prebid_auc": "old AUC", "old_prebid_brier": "old Brier", "old_prebid_acc": "old accuracy"})
    st.dataframe(t.style.format({"new AUC": "{:.2f}", "new Brier": "{:.3f}", "new accuracy": "{:.0%}", "old AUC": "{:.2f}", "old Brier": "{:.3f}", "old accuracy": "{:.0%}"}), width="stretch")
    rule = wf.get("prebid_rule")
    if rule:
        st.caption(f"Why the 80% rule: pooled out-of-sample, only {rule['n_high']} pre-bid probabilities reached 80% and {rule['hit_high']:.0%} of those listed above issue; "
                   f"the other {rule['n_low']} listed above issue {rule['hit_low']:.0%} of the time against a {rule['base_rate']:.0%} base rate, so below 80% the app says 'about the base rate'.")
    fq = wf.get("forecast")
    if fq is not None and len(fq):
        st.markdown("**The QIB forecast behind it** (log1p of final QIB subscription)")
        st.dataframe(fq.rename(columns={"n": "IPOs", "rmse": "error (RMSE)", "r2_vs_mean": "R² vs average", "corr": "correlation"}).style.format(
            {"IPOs": "{:.0f}", "error (RMSE)": "{:.2f}", "R² vs average": "{:.2f}", "correlation": "{:.2f}"}), width="stretch")
        st.caption("R² near 0.1 means fundamentals explain little of how much QIB demand an IPO draws, which is why pre-bid ranges are wide.")
    names = {"A": "Model A (quantile regression, final figures)", "B": "Model B (similar QIB, final figures)",
             "prebid": "Pre-bid, new (QIB forecast)", "prebid_knn": "Pre-bid, old (10 nearest IPOs)"}
    g = wf["gain"].reset_index()
    g["model"] = g["model"].map(names)
    g = g.rename(columns={"period": "test year", "mae_p50": "median error (pts)", "cov_25_75_raw": "25-75 covers (raw)", "cov_10_90_raw": "10-90 covers (raw)",
                          "cov_25_75_cal": "25-75 covers (widened)", "cov_10_90_cal": "10-90 covers (widened)"})
    pct = {c: "{:.0%}" for c in g.columns if "covers" in c}
    st.markdown("**Expected listing gain**")
    st.dataframe(g.style.format(pct | {"n": "{:.0f}", "median error (pts)": "{:.1f}", "spearman": "{:.2f}"}), hide_index=True, width="stretch")
    st.caption("Targets: 25-75 should cover about 50%, 10-90 about 80%. 'Widened' ranges use only the other test years' residuals, so the "
               "widening is never scored on the data it was fitted on.")
    st.markdown("**Isotonic calibration of the subscription model's probabilities, before and after** (each year calibrated on the other years only)")
    cmp_ = pr[["demand_brier_raw", "demand_brier_cal", "demand_ece_raw", "demand_ece_cal", "demand_acc_raw", "demand_acc_cal"]].rename(columns={
        "demand_brier_raw": "Brier before", "demand_brier_cal": "Brier after", "demand_ece_raw": "calibration error before",
        "demand_ece_cal": "calibration error after", "demand_acc_raw": "accuracy before", "demand_acc_cal": "accuracy after"})
    st.dataframe(cmp_.style.format({"Brier before": "{:.3f}", "Brier after": "{:.3f}", "calibration error before": "{:.3f}", "calibration error after": "{:.3f}",
                                    "accuracy before": "{:.0%}", "accuracy after": "{:.0%}"}), width="stretch")
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "calibration_before", "Pooled calibration before (deciles)"), (c2, "calibration_after", "Pooled calibration after (deciles)")):
        col.markdown(title)
        col.dataframe(wf[key].style.format({"n": "{:.0f}", "predicted": "{:.0%}", "actual": "{:.0%}"}), width="stretch")
    st.caption("Isotonic calibration is only applied if it lowers the walk-forward Brier score by at least 0.005. With a few hundred IPOs it can overfit, "
               "and here it does not clear that bar, so the app shows the subscription model's probabilities as fitted. Change `ISOTONIC` in src/report.py to force it on.")


def show_backtest(bt: dict, wf: dict | None = None) -> None:
    show_walk_forward(wf)
    if not bt.get("demand"):
        st.info("Not enough 2026 rows to score.")
        return
    st.markdown("### Single test: trained 2022-2025, scored on 2026")
    st.markdown(f"**{bt['n_test']} IPOs listed in 2026. These models were fitted on {bt['n_train']} IPOs from 2022-2025 and never saw them.** "
                "It is one year, so the walk-forward above is the fairer summary.")
    rows = {"Subscription model (final figures)": bt["demand"], "Pre-bid, new (QIB forecast)": bt["prebid"], "Pre-bid, old (logistic on fundamentals)": bt["prebid_old"]}
    st.dataframe(pd.DataFrame(rows).T[["n", "auc", "brier", "accuracy", "base_rate"]].style.format({"n": "{:.0f}", "auc": "{:.2f}", "brier": "{:.3f}", "accuracy": "{:.0%}", "base_rate": "{:.0%}"}), width="stretch")
    st.dataframe(bt["baselines"].style.format("{:.0%}"), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("Calibration, subscription model (deciles)")
    c1.dataframe(bt["calibration_demand_raw"].style.format({"predicted": "{:.0%}", "actual": "{:.0%}"}), width="stretch")
    c2.markdown("Calibration, pre-bid model (deciles)")
    c2.dataframe(bt["calibration_prebid"].style.format({"predicted": "{:.0%}", "actual": "{:.0%}"}), width="stretch")
    g = pd.DataFrame({"Model A (quantile regression)": bt["gain_a"], "Model B (similar QIB)": bt["gain_b"],
                      "Pre-bid, new (QIB forecast)": bt["gain_prebid"], "Pre-bid, old (10 nearest IPOs)": bt["gain_prebid_knn"]}).T
    st.markdown(f"Listing-gain %: Model {bt['gain_model_default']} is the default (chosen on the last full year inside the training data). "
                "Ranges are widened using 2024-2025 walk-forward residuals only.")
    st.dataframe(g.style.format({"n": "{:.0f}", "mae_p50": "{:.1f}", "cov_25_75": "{:.0%}", "cov_10_90": "{:.0%}", "cov_25_75_raw": "{:.0%}", "cov_10_90_raw": "{:.0%}",
                                 "spearman": "{:.2f}", "pinball": "{:.2f}"}), width="stretch")
    st.caption("mae_p50 = median absolute error of the 50th-percentile prediction, in percentage points. cov_* = share of outcomes inside the range (targets 50% and 80%); "
               "the _raw columns are before widening.")
    if bt.get("widen", {}).get("A"):
        w = pd.DataFrame({m: {band: f"+{v['inner']:.1f} / +{v['outer']:.1f}" for band, v in tb.items()} for m, tb in bt["widen"].items()}).T
        st.markdown("Range widening by QIB band (points added to each side, inner 25-75 / outer 10-90)")
        st.dataframe(w.rename(index={"A": "Model A", "B": "Model B", "prebid": "Pre-bid (QIB forecast)", "prebid_knn": "Pre-bid (10 nearest)"}), width="stretch")
    st.markdown("Strategies on 2026")
    st.dataframe(bt["strategies"].rename(columns={"expected_gain_per_application_pct": "expected gain per application %"}).style.format(
        {"hit_rate": "{:.0%}", "avg_gain_pct": "{:+.1f}", "expected gain per application %": "{:+.1f}", "sum_expected_gain_per_lot_rs": "₹{:,.0f}"}), width="stretch")
    st.caption("Expected gain per application = chance of allotment (about 1 / retail subscription, capped at 1) × the listing gain.")


if page == "Analyse a new IPO":
    page_new()
    st.stop()
if page == "Model accuracy":
    st.header("Model accuracy")
    _d, _pub, _prod, _bt, _wf = report_models((ROOT / "data" / "ipos.csv").stat().st_mtime)
    show_backtest(_bt, _wf)
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


{"Past IPOs": page_dataset, "Winners vs losers": page_groups,
 "Strategy test": page_strategy, "Check a past IPO": page_check}[page]()
