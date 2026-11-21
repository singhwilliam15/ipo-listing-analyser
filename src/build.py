"""Pipeline CLI. Run from the ipo-analyser folder:

    python -m src.build issues                 # NSE list of past issues -> data/issues.csv
    python -m src.build outcomes               # listing-day prices      -> data/outcomes.csv
    python -m src.build demand                 # NSE bid data + RHP links -> data/demand.csv
    python -m src.build market                 # Nifty + India VIX       -> data/market.csv
    python -m src.build rhp                    # download RHP PDFs       -> rhp/<symbol>.pdf
    python -m src.build extract --limit 10     # Claude reads RHPs       -> data/raw/<symbol>.json
    python -m src.build assemble               # everything              -> data/ipos.csv, data/ipos_sme.csv
    python -m src.build analyse                # report                  -> data/report.md

Every step is resumable: rows already fetched are skipped unless you pass --refresh.
Hand corrections go in data/manual/overrides.csv (symbol,field,value,note) and win over
every automatic source.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import analyse, extract_rhp, features, schema
from .fetch_market import DataError, NSEClient, find_rhp_url, index_closes, issue_info, listing_outcome, parse_bids

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ISSUES, OUTCOMES, DEMAND, MARKET = (DATA / f for f in ("issues.csv", "outcomes.csv", "demand.csv", "market.csv"))
OVERRIDES = DATA / "manual" / "overrides.csv"
WARMUP_START = date(2021, 10, 1)    # a few months before training so section F has history


def log(msg: str) -> None:
    print(msg, flush=True)


def read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


def issues_frame(segment: str | None = "Mainboard") -> pd.DataFrame:
    df = read(ISSUES, parse_dates=["issue_open", "issue_close", "listing_date"])
    if df.empty:
        sys.exit("data/issues.csv not found. Run `python -m src.build issues` first.")
    df = apply_overrides(df, fields=("issue_price", "price_low", "price_high", "lot_size", "listing_date", "issue_open", "issue_close", "segment"))
    return df if segment is None else df[df["segment"] == segment]


def overrides() -> pd.DataFrame:
    if not OVERRIDES.exists():
        return pd.DataFrame(columns=["symbol", "field", "value", "note"])
    import io
    lines = [l for l in OVERRIDES.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)), dtype=str).dropna(subset=["symbol", "field"])


def apply_overrides(df: pd.DataFrame, fields=None) -> pd.DataFrame:
    ov = overrides()
    if fields is not None:
        ov = ov[ov["field"].isin(fields)]
    if ov.empty:
        return df
    df = df.copy()
    for r in ov.itertuples():
        m = df["symbol"] == r.symbol.strip().upper()
        if not m.any():
            continue
        col = schema.BY_NAME.get(r.field)
        val = r.value
        if col is not None and col.dtype == "date":
            val = pd.to_datetime(val, dayfirst=True, errors="coerce")
        elif col is not None and col.dtype in ("number", "integer", "flag"):
            val = pd.to_numeric(val, errors="coerce")
        if r.field not in df:
            df[r.field] = pd.Series(None, index=df.index, dtype=object)
        df.loc[m, r.field] = val
    return df


def upsert(path: Path, new: pd.DataFrame, key: str = "symbol") -> pd.DataFrame:
    old = read(path)
    out = pd.concat([old[~old[key].isin(new[key])] if len(old) else old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def cmd_issues(a) -> None:
    if a.csv:
        df = pd.read_csv(a.csv)
        need = {"symbol", "company", "issue_close", "listing_date", "issue_price"}
        if need - set(df.columns):
            sys.exit(f"{a.csv} is missing columns: {sorted(need - set(df.columns))}")
        if "segment" not in df:
            df["segment"] = "Mainboard"
        for c in ("issue_open", "issue_close", "listing_date"):
            if c in df:
                df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
        for c in ("price_low", "price_high"):
            if c not in df:
                df[c] = df["issue_price"]
    else:
        df = NSEClient().past_issues(date.fromisoformat(a.start), date.fromisoformat(a.end))
    if df.empty:
        sys.exit("No issues returned.")
    upsert(ISSUES, df)
    log(f"{len(df)} issues saved to {ISSUES.relative_to(ROOT)} "
        f"({(df['segment'] == 'Mainboard').sum()} mainboard, {(df['segment'] == 'SME').sum()} SME).")


def cmd_outcomes(a) -> None:
    iss = issues_frame(None if a.all_segments else "Mainboard")
    done = set(read(OUTCOMES).get("symbol", pd.Series(dtype=str))) if not a.refresh else set()
    todo = iss[~iss["symbol"].isin(done) & (iss["listing_date"] <= pd.Timestamp.today())]
    nse = NSEClient()
    log(f"Pricing {len(todo)} listings (NSE, Yahoo fallback)…")
    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        o = listing_outcome(r.symbol, r.listing_date, float(r.issue_price), nse)
        rows.append({"symbol": r.symbol, **o})
        if i % 20 == 0 or i == len(todo):
            upsert(OUTCOMES, pd.DataFrame(rows))
            rows = []
            log(f"  {i}/{len(todo)}")
    if rows:
        upsert(OUTCOMES, pd.DataFrame(rows))


def cmd_demand(a) -> None:
    if a.reparse:                                   # rebuild from cached NSE responses, no network
        rows = []
        for f in sorted((DATA / "nse_detail").glob("*.json")):
            det = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"symbol": f.stem, **parse_bids(det), **issue_info(det), "rhp_url": find_rhp_url(det), "note": ""})
        pd.DataFrame(rows).to_csv(DEMAND, index=False)
        log(f"Re-parsed {len(rows)} cached NSE detail files into {DEMAND.relative_to(ROOT)}.")
        return
    iss = issues_frame(None if a.all_segments else "Mainboard")
    done = set(read(DEMAND).get("symbol", pd.Series(dtype=str))) if not a.refresh else set()
    todo = iss[~iss["symbol"].isin(done)]
    nse = NSEClient()
    log(f"Fetching bid data for {len(todo)} issues from NSE…")
    rows, raw_dir = [], DATA / "nse_detail"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            det = nse.ipo_detail(r.symbol, getattr(r, "series", "EQ") or "EQ")
            (raw_dir / f"{r.symbol}.json").write_text(json.dumps(det), encoding="utf-8")
            rows.append({"symbol": r.symbol, **parse_bids(det), **issue_info(det), "rhp_url": find_rhp_url(det), "note": ""})
        except DataError as exc:
            rows.append({"symbol": r.symbol, "note": str(exc)})
        if i % 20 == 0 or i == len(todo):
            upsert(DEMAND, pd.DataFrame(rows))
            rows = []
            log(f"  {i}/{len(todo)}")


def cmd_market(a) -> None:
    m = index_closes(WARMUP_START - timedelta(days=120), date.today())
    m.index.name = "date"
    MARKET.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(MARKET)
    log(f"{len(m)} days of Nifty / VIX saved to {MARKET.relative_to(ROOT)}.")


def _targets(a, segment="Mainboard") -> pd.DataFrame:
    iss = issues_frame(segment)
    iss = iss[iss["listing_date"] >= pd.Timestamp(schema.TRAIN_START)]
    if a.year:
        iss = iss[iss["listing_date"].dt.year == a.year]
    if a.symbols:
        iss = iss[iss["symbol"].isin([s.upper() for s in a.symbols])]
    return iss


def cmd_rhp(a) -> None:
    dem = read(DEMAND)
    urls = dict(zip(dem["symbol"], dem["rhp_url"])) if "rhp_url" in dem else {}
    ov = overrides()
    urls.update({s.strip().upper(): v for s, f, v in zip(ov["symbol"], ov["field"], ov["value"]) if f == "rhp_url"})
    nse, missing = NSEClient(), []
    extract_rhp.RHP_DIR.mkdir(exist_ok=True)
    for r in _targets(a).itertuples():
        pdf = extract_rhp.RHP_DIR / f"{r.symbol}.pdf"
        if pdf.exists():
            continue
        url = urls.get(r.symbol)
        if not isinstance(url, str) or not url:
            missing.append(r.symbol)
            continue
        try:
            nse.download(url, pdf)
            log(f"  downloaded {r.symbol}")
        except DataError as exc:
            log(f"  {r.symbol}: {exc}")
            missing.append(r.symbol)
    if missing:
        log(f"\n{len(missing)} issues have no RHP yet. Put the PDF at {extract_rhp.RHP_DIR / '<SYMBOL>.pdf'}, or add a row "
            f"'SYMBOL,rhp_url,<link>' to data/manual/overrides.csv (SEBI: sebi.gov.in > Filings > Public Issues).")
        (DATA / "missing_rhp.txt").write_text("\n".join(missing), encoding="utf-8")


def cmd_extract(a) -> None:
    """Extract RHPs; with --wait, sleep through plan usage limits and carry on until every RHP is done."""
    import time
    a.run_start = time.time()
    while _extract_round(a) and a.wait:
        log(f"Waiting {a.wait} min for the usage limit to reset...")
        time.sleep(a.wait * 60)


def _extract_round(a) -> bool:
    """One pass over the pending RHPs. Returns True if it stopped on a rate/usage limit."""
    def pending(sym):
        out = extract_rhp.RAW_DIR / f"{sym}.json"
        if not out.exists():
            return True
        return a.force and out.stat().st_mtime < a.run_start      # --force: redo once per run
    cands = _targets(a).dropna(subset=["listing_date"]).sort_values("listing_date", ascending=False)   # newest first
    cands = [r.symbol for r in cands.itertuples() if (extract_rhp.RHP_DIR / f"{r.symbol}.pdf").exists()]
    if a.max_total:
        have = [s for s in cands if not pending(s)]           # already extracted: sunk cost, always kept
        new = [s for s in cands if pending(s)][: max(0, a.max_total - len(have))]   # then the newest pending ones
        cands = have + new
    todo = [s for s in cands if pending(s)]
    if not todo:
        log("Nothing left to extract.")
        return False
    if a.limit:
        todo = todo[: a.limit]
    if a.estimate:
        tot = 0
        for s in todo:
            pages = extract_rhp.page_texts(extract_rhp.RHP_DIR / f"{s}.pdf")
            tot += extract_rhp.estimate_tokens(extract_rhp.build_prompt(pages, extract_rhp.select_pages(pages)))
        cost = tot / 1e6 * 4 + len(todo) * 5000 / 1e6 * 20     # Opus 5.5: $4/M in, $20/M out, ~5k out tokens each
        log(f"{len(todo)} RHPs, ~{tot:,} input tokens, roughly ${cost:,.2f} (estimate).")
        return False
    import os
    import anthropic
    backend = a.backend
    if backend == "auto":
        backend = "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
    client = anthropic.Anthropic() if backend == "api" else extract_rhp.ClaudeCLI()
    log(f"Extracting {len(todo)} RHPs via {backend}.")
    spent_in = spent_out = 0
    limited = False

    def one(s):
        return s, extract_rhp.extract(s, force=a.force, client=client)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    pool = ThreadPoolExecutor(max_workers=max(1, a.workers))
    futures = [pool.submit(one, s) for s in todo]
    for i, fut in enumerate(as_completed(futures), 1):
        try:
            s, rec = fut.result()
            spent_in += rec["usage"]["input_tokens"]
            spent_out += rec["usage"]["output_tokens"]
            flag = f"  ⚠ {len(rec['checks'])} check(s): {'; '.join(rec['checks'])}" if rec["checks"] else ""
            log(f"[{i}/{len(todo)}] {s}: {len(rec['pages_sent'])} pages sent{flag}")
        except (anthropic.RateLimitError, extract_rhp.RateLimited) as exc:
            log(f"Rate/usage limit reached ({exc}).")
            limited = True
            for f in futures:
                f.cancel()
            break
        except Exception as exc:                       # one bad RHP must not stop the run
            sym = todo[futures.index(fut)]
            log(f"[{i}/{len(todo)}] {sym}: FAILED – {type(exc).__name__}: {str(exc)[:300]}")
    pool.shutdown(wait=True, cancel_futures=True)
    log(f"Tokens this round: {spent_in:,} in, {spent_out:,} out (≈ ${spent_in / 1e6 * 4 + spent_out / 1e6 * 20:,.2f} at API prices).")
    return limited


def load_raw() -> pd.DataFrame:
    rows = []
    for p in sorted(extract_rhp.RAW_DIR.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        fields = rec["fields"]
        if rec.get("extract"):                     # re-derive with the current code, no new API call
            try:
                fields = extract_rhp.flatten(extract_rhp.RHPExtract.model_validate(rec["extract"]))
            except Exception as exc:
                log(f"{rec['symbol']}: could not re-flatten ({exc}); using stored fields")
        f = {k: v for k, v in fields.items() if not k.startswith("rhp_page_")}
        rows.append({"symbol": rec["symbol"], **f, "has_rhp": True, "rhp_check_count": len(rec.get("checks", []))})
    return pd.DataFrame(rows)


def assemble_segment(segment: str) -> pd.DataFrame:
    iss = issues_frame(segment)
    df = iss.copy()
    for extra in (read(OUTCOMES), read(DEMAND)):
        if len(extra):
            df = df.merge(extra.drop(columns=[c for c in ("note",) if c in extra]), on="symbol", how="left")
    raw = load_raw()
    if len(raw):
        raw = raw.drop(columns=[c for c in ("company_rhp",) if c in raw])
        # NSE wins for price/lot; RHP fills gaps
        dup = [c for c in raw.columns if c in df.columns and c != "symbol"]
        df = df.merge(raw, on="symbol", how="left", suffixes=("", "_rhp"))
        for c in dup:
            df[c] = df[c].where(df[c].notna(), df[f"{c}_rhp"])
            df = df.drop(columns=f"{c}_rhp")
    for c in ("lot_size", "lead_managers"):          # NSE issue page fills what the list/RHP lack
        if f"{c}_nse" in df:
            df[c] = df[c].where(df[c].notna(), df[f"{c}_nse"]) if c in df else df[f"{c}_nse"]
    df = apply_overrides(df)
    df = features.derive(df)
    market = read(MARKET, index_col="date", parse_dates=["date"])
    df = features.market_context(df, market, history=df[["listing_date", "listing_gain_pct"]])
    df["data_quality"] = features.data_quality(df)
    df = df[df["listing_date"] >= pd.Timestamp(schema.TRAIN_START)]
    return schema.conform(df).sort_values("listing_date").reset_index(drop=True)


def cmd_assemble(a) -> None:
    for seg, name in (("Mainboard", "ipos.csv"), ("SME", "ipos_sme.csv")):
        if not (issues_frame(None)["segment"] == seg).any():
            continue
        df = assemble_segment(seg)
        df.to_csv(DATA / name, index=False)
        q = df["data_quality"].value_counts().to_dict()
        br = analyse.base_rate(analyse.add_split(df).query("split == 'train'"))
        log(f"{name}: {len(df)} rows, quality {q}. "
            f"Base rate (train): {br['positive']}/{br['n']} = {br['rate'] * 100:.1f}% listed above issue price.")


def cmd_analyse(a) -> None:
    df = analyse.load(DATA / ("ipos_sme.csv" if a.sme else "ipos.csv"))
    res = analyse.evaluate_all(df, target=a.target, quality=tuple(a.quality))
    md = analyse.report_markdown(res)
    out = DATA / f"report_{a.target}{'_sme' if a.sme else ''}.md"
    out.write_text(md, encoding="utf-8")
    log(md)
    log(f"\nSaved to {out.relative_to(ROOT)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.build")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("issues")
    p.add_argument("--start", default=WARMUP_START.isoformat())
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--csv", help="import a hand-made issue list instead of calling NSE")
    for name in ("outcomes", "demand"):
        p = sub.add_parser(name)
        p.add_argument("--refresh", action="store_true")
        p.add_argument("--all-segments", action="store_true", help="include SME issues")
        if name == "demand":
            p.add_argument("--reparse", action="store_true", help="rebuild from data/nse_detail/*.json without calling NSE")
    sub.add_parser("market")
    for name in ("rhp", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--symbols", nargs="*")
        p.add_argument("--year", type=int)
        if name == "extract":
            p.add_argument("--limit", type=int)
            p.add_argument("--force", action="store_true")
            p.add_argument("--max-total", type=int, help="only the N most recently listed RHPs (already-extracted ones count)")
            p.add_argument("--estimate", action="store_true", help="estimate tokens and cost, no API calls")
            p.add_argument("--backend", default="auto", choices=["auto", "api", "cli"],
                           help="api = ANTHROPIC_API_KEY; cli = Claude Code login (`claude -p`); auto picks api if a key is set")
            p.add_argument("--workers", type=int, default=1, help="parallel extractions")
            p.add_argument("--wait", type=int, default=0, metavar="MIN",
                           help="on a usage limit, sleep MIN minutes and resume (0 = stop)")
    sub.add_parser("assemble")
    p = sub.add_parser("analyse")
    p.add_argument("--target", default="target_gain", choices=analyse.TARGETS)
    p.add_argument("--quality", nargs="+", default=["full", "partial"])
    p.add_argument("--sme", action="store_true")
    a = ap.parse_args(argv)
    {"issues": cmd_issues, "outcomes": cmd_outcomes, "demand": cmd_demand, "market": cmd_market,
     "rhp": cmd_rhp, "extract": cmd_extract, "assemble": cmd_assemble, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
