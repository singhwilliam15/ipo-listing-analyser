"""Public market data: NSE issue lists and bid data, Yahoo Finance prices, Nifty and India VIX.

NSE's JSON API is undocumented and its field names drift, so every parser here looks
fields up by a list of aliases and leaves a value empty rather than guessing.
NSE blocks many cloud/VPN IP ranges; run this from a normal home connection in India.
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

NSE = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


class DataError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# NSE client
# --------------------------------------------------------------------------- #
class NSEClient:
    def __init__(self, pause: float = 0.7, timeout: float = 20.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                               "Accept-Language": "en-IN,en;q=0.9", "Referer": f"{NSE}/market-data/all-upcoming-issues-ipo"})
        self.pause, self.timeout, self._warm = pause, timeout, False

    def _warmup(self) -> None:
        if not self._warm:
            try:
                self.s.get(f"{NSE}/market-data/all-upcoming-issues-ipo", timeout=self.timeout)
            except requests.RequestException as exc:
                raise DataError(f"NSE unreachable: {exc}") from exc
            self._warm = True

    def get_json(self, path: str, **params) -> Any:
        self._warmup()
        for attempt in range(3):
            time.sleep(self.pause)
            try:
                r = self.s.get(f"{NSE}/api/{path}", params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                err = exc
            else:
                if r.status_code == 200 and r.text.strip():
                    try:
                        return r.json()
                    except ValueError as exc:
                        err = exc
                else:
                    err = DataError(f"HTTP {r.status_code}")
                if r.status_code in (401, 403):     # cookie expired: re-warm once
                    self._warm = False
                    self._warmup()
        raise DataError(f"NSE {path} failed: {err}")

    def download(self, url: str, dest) -> None:
        """Save the RHP PDF at `dest`. NSE serves a zip holding the RHP plus addenda/GID; keep the RHP."""
        import zipfile
        self._warmup()
        err = None
        for _ in range(2):                       # one retry: large zips sometimes arrive truncated
            try:
                r = self.s.get(url, timeout=300, headers={"Accept": "application/pdf,application/zip,*/*"})
            except requests.RequestException as exc:
                err = exc
                continue
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
                continue
            try:
                if r.content.startswith(b"%PDF"):
                    dest.write_bytes(r.content)
                    return
                if r.content.startswith(b"PK"):
                    dest.write_bytes(pick_rhp_from_zip(r.content))
                    return
                err = "not a PDF or zip"
            except zipfile.BadZipFile:
                err = "corrupt zip"
        raise DataError(f"download failed ({err}) for {url}")


    # -- endpoints --------------------------------------------------------- #
    def past_issues(self, start: date, end: date) -> pd.DataFrame:
        """All issues (mainboard + SME) that listed between start and end, fetched month by month
        (a yearly window silently dropped some issues)."""
        rows: list[dict] = []
        cur = start
        while cur <= end:
            nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
            stop = min(end, nxt - timedelta(days=1))
            js = self.get_json("public-past-issues", from_date=cur.strftime("%d-%m-%Y"),
                               to_date=stop.strftime("%d-%m-%Y"), security_type="all")
            rows += js if isinstance(js, list) else js.get("data", [])
            cur = stop + timedelta(days=1)
        return parse_past_issues(rows)

    def ipo_detail(self, symbol: str, series: str = "EQ") -> Any:
        return self.get_json("ipo-detail", symbol=symbol, series=series)

    def history(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily OHLC as traded (never adjusted for splits/bonuses), all series."""
        js = self.get_json("historicalOR/generateSecurityWiseHistoricalData", symbol=symbol,
                           type="priceVolumeDeliverable", series="ALL",
                           **{"from": start.strftime("%d-%m-%Y"), "to": end.strftime("%d-%m-%Y")})
        return parse_nse_history(js.get("data", []) if isinstance(js, dict) else js)


SERIES_RANK = {"EQ": 0, "BE": 1, "BZ": 2, "SM": 3, "ST": 4}


def parse_nse_history(rows: Iterable[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        day = pd.to_datetime(pick(r, "mTIMESTAMP", "mtimestamp"), errors="coerce")
        if pd.isna(day):
            continue
        recs.append({"date": day.normalize(), "series": pick(r, "CH_SERIES", "chSeries") or "",
                     "Open": num(pick(r, "CH_OPENING_PRICE", "chOpeningPrice")),
                     "High": num(pick(r, "CH_TRADE_HIGH_PRICE", "chTradeHighPrice")),
                     "Low": num(pick(r, "CH_TRADE_LOW_PRICE", "chTradeLowPrice")),
                     "Close": num(pick(r, "CH_CLOSING_PRICE", "chClosingPrice"))})
    if not recs:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.DataFrame(recs)
    df["rank"] = df["series"].map(SERIES_RANK).fillna(9)
    df = df.sort_values(["date", "rank"]).drop_duplicates("date").set_index("date")
    return df[["Open", "High", "Low", "Close"]]


def pick_rhp_from_zip(blob: bytes) -> bytes:
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(blob))
    pdfs = [i for i in z.infolist() if i.filename.lower().endswith(".pdf")]
    if not pdfs:
        raise DataError("zip has no PDF")

    def rank(i):
        n = i.filename.lower()
        named = "red herring" in n or re.search(r"\brhp\b", n) is not None
        noise = any(w in n for w in ("corrigendum", "addendum", "gid", "general information", "letter"))
        return (named and not noise, not noise, i.file_size)
    best = max(pdfs, key=rank)
    if not rank(best)[1]:
        raise DataError("zip holds only addenda / GID, no RHP: " + ", ".join(i.filename for i in pdfs))
    return z.read(best)


# --------------------------------------------------------------------------- #
# Parsers (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def pick(d: dict, *aliases: str):
    """First non-empty value whose key matches any alias (case/space/underscore-insensitive)."""
    norm = {re.sub(r"[\s_\-]", "", k).lower(): v for k, v in d.items()}
    for a in aliases:
        v = norm.get(re.sub(r"[\s_\-]", "", a).lower())
        if v not in (None, "", "-", "NA", "N/A"):
            return v
    return None


def num(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v))
    return float(m.group().replace(",", "")) if m else np.nan


def price_band(v) -> tuple[float, float]:
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", str(v or ""))]
    if not nums:
        return np.nan, np.nan
    return min(nums), max(nums)


def parse_past_issues(rows: Iterable[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        sym = pick(r, "symbol")
        if not sym:
            continue
        lo, hi = price_band(pick(r, "priceRange", "issuePriceBand", "priceBand", "price_range"))
        price = num(pick(r, "issuePrice", "issue_price", "offerPrice"))
        sec = str(pick(r, "securityType", "series", "security_type", "issueType") or "")
        if sec and not re.fullmatch(r"\s*(equity|sme|eq|sm|st)\s*", sec, re.I):
            continue                     # DEBT (NCDs), InvIT, REIT are not equity IPOs
        if re.search(r"PP\d*$", str(sym).strip().upper()):
            continue                     # partly-paid shares from a rights issue (e.g. ADANIENPP1)
        out.append({
            "symbol": str(sym).strip().upper(),
            "company": pick(r, "companyName", "company", "company_name", "name"),
            "segment": "SME" if re.search(r"sme|\bsm\b|\bst\b", sec, re.I) else "Mainboard",
            "issue_open": pd.to_datetime(pick(r, "ipoStartDate", "issueStartDate", "issue_start_date", "openDate"), errors="coerce", dayfirst=True),
            "issue_close": pd.to_datetime(pick(r, "ipoEndDate", "issueEndDate", "issue_end_date", "closeDate"), errors="coerce", dayfirst=True),
            "listing_date": pd.to_datetime(pick(r, "listingDate", "listing_date", "dateOfListing"), errors="coerce", dayfirst=True),
            "price_low": lo,
            "price_high": hi if not np.isnan(hi) else price,
            "issue_price": price if not np.isnan(price) else hi,
            "lot_size": num(pick(r, "lotSize", "marketLot", "minBidQuantity", "lot_size")),
            "series": "SM" if "SME" in sec.upper() else "EQ",
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.drop_duplicates("symbol", keep="last").sort_values("listing_date").reset_index(drop=True)


def issue_row_from_detail(symbol: str, detail: Any) -> dict | None:
    """One past-issues row built from an ipo-detail payload. NSE's past-issues list omits some
    mainboard IPOs (Ola Electric, IREDA, NTPC Green, Tata Capital, Aegis Vopak) though their issue pages work."""
    if not isinstance(detail, dict) or not detail.get("metaInfo"):
        return None
    meta, rows = detail["metaInfo"], {}
    for d in _walk(detail.get("issueInfo", {})):
        if isinstance(d.get("title"), str) and "value" in d:
            rows[d["title"].strip().lower()] = str(d.get("value") or "").strip().strip('"')
    period = re.findall(r"\d{1,2}-[A-Za-z]{3}-\d{4}", rows.get("issue period", ""))
    lo, hi = price_band(rows.get("price range"))
    lot = num(re.findall(r"[\d,]+", rows.get("bid lot", ""))[0]) if re.findall(r"[\d,]+", rows.get("bid lot", "")) else np.nan
    if len(period) < 2 or not meta.get("listingDate") or np.isnan(hi):
        return None
    return {"symbol": symbol.upper(), "company": meta.get("companyName"), "segment": "Mainboard",
            "issue_open": pd.to_datetime(period[0]), "issue_close": pd.to_datetime(period[1]),
            "listing_date": pd.to_datetime(meta["listingDate"]), "price_low": lo, "price_high": hi,
            "issue_price": hi, "lot_size": lot, "series": "EQ"}


CATEGORY_PATTERNS = [   # order matters: sNII/bNII before generic NII
    ("sub_qib_x", r"qualified|\bqib"),
    ("sub_snii_x", r"snii|small|two lakh|2\s*lakh|(below|upto|up to|less than)\s*(rs\.?|₹)?\s*10|2\s*-\s*10"),
    ("sub_bnii_x", r"bnii|big|(above|more than)\s*(rs\.?|₹)?\s*(10|ten)\b"),
    ("sub_nii_x", r"non[\s\-]*institutional|\bnii\b|\bhni\b"),
    ("sub_retail_x", r"retail|\brii\b|individual"),
    ("sub_employee_x", r"employee"),
    ("sub_total_x", r"^\s*total|overall|grand total"),
]


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def parse_bids(detail: Any) -> dict:
    """Category-wise subscription (times) and application count from an ipo-detail payload.

    NSE's `activeCat` block is consolidated across NSE + BSE; `bidDetails` is NSE-only, so it
    is used only when `activeCat` is missing."""
    src = detail.get("activeCat") if isinstance(detail, dict) and detail.get("activeCat") else detail
    res: dict[str, float] = {}
    for d in _walk(src):
        cat = pick(d, "category", "categoryName", "investorCategory", "srNoCategory")
        times = pick(d, "noOfTotalMeant", "noOfTimes", "noOfTimesSubscribed", "times", "subscription", "noOfTime")
        if cat is None or times is None:
            continue
        label = str(cat)
        if re.search(r"anchor|shareholder|policy", label, re.I):
            continue
        for col, pat in CATEGORY_PATTERNS:
            if re.search(pat, label, re.I):
                res.setdefault(col, num(times))   # first row for a category wins
                break
    apps = None
    for d in _walk(detail):
        apps = pick(d, "noOfApplications", "totalApplications", "applications")
        if apps is not None:
            break
    res["n_applications"] = num(apps)
    return res


def issue_info(detail: Any) -> dict:
    """Fields from the title/value list in ipo-detail -> issueInfo."""
    rows = {}
    for d in _walk(detail.get("issueInfo", {}) if isinstance(detail, dict) else {}):
        if isinstance(d.get("title"), str) and "value" in d:
            rows[d["title"].strip().lower()] = str(d.get("value") or "").strip().strip('"')
    out = {"lot_size_nse": np.nan, "lead_managers_nse": None, "anchor_shares_cr": np.nan}
    lot = rows.get("bid lot") or rows.get("minimum order quantity")
    if lot:
        out["lot_size_nse"] = num(lot)
    lm = rows.get("book running lead managers") or rows.get("book running lead manager")
    if lm:
        names = [n.strip(" .") for n in re.split(r",|\band\b", lm) if n.strip(" .")]
        out["lead_managers_nse"] = "; ".join(names)
    m = re.search(r"anchor\s*(?:investors?)?\s*(?:portion|allocation)\s*(?:of)?\s*(?:up\s*to\s+)?([\d,]{4,})",
                  rows.get("issue size", ""), re.I)
    if m:
        out["anchor_shares_cr"] = float(m.group(1).replace(",", "")) / 1e7
    return out


def find_rhp_url(detail: Any) -> str | None:
    """The RHP link in an ipo-detail payload (NSE gives a zip), or None."""
    for d in _walk(detail):
        t, v = d.get("title"), d.get("value")
        if isinstance(t, str) and isinstance(v, str) and "red herring" in t.lower():
            m = re.search(r"https?://\S+?\.(?:zip|pdf)", v, re.I)
            if m:
                return m.group()
    best = None
    for d in _walk(detail):
        for k, v in d.items():
            if isinstance(v, str) and v.lower().split("?")[0].endswith(".pdf"):
                label = f"{k} {d.get('title', '')} {d.get('name', '')} {v}".lower()
                if "red herring" in label or "rhp" in label:
                    return v if v.startswith("http") else NSE + v
                if best is None and "prospectus" in label:
                    best = v if v.startswith("http") else NSE + v
    return best


# --------------------------------------------------------------------------- #
# Yahoo Finance prices
# --------------------------------------------------------------------------- #
def _yf():
    import yfinance as yf
    return yf


def unadjusted_history(symbol: str, start: pd.Timestamp, days: int = 50) -> pd.DataFrame:
    """Daily OHLC from `start`, with later splits/bonuses undone so prices compare to the issue price."""
    yf = _yf()
    for suffix in (".NS", ".BO"):
        tk = yf.Ticker(symbol + suffix)
        h = tk.history(start=start, end=start + pd.Timedelta(days=days), auto_adjust=False, actions=True)
        if h.empty:
            continue
        h.index = h.index.tz_localize(None).normalize()
        splits = tk.splits
        if splits is not None and len(splits):
            splits.index = splits.index.tz_localize(None).normalize()
            later = splits[splits.index > h.index[0]]
            factor = float(np.prod(later.values)) if len(later) else 1.0
            # yfinance back-adjusts for splits; multiply prices on each day by the splits after that day
            for i, day in enumerate(h.index):
                f = float(np.prod(later[later.index > day].values)) if len(later) else 1.0
                h.iloc[i, h.columns.get_indexer(["Open", "High", "Low", "Close"])] *= f
            h.attrs["split_factor_listing"] = factor
        return h[["Open", "High", "Low", "Close"]]
    return pd.DataFrame()


def listing_outcome(symbol: str, listing_date, issue_price: float, nse: "NSEClient | None" = None) -> dict:
    """Listing-day OHLC and 1-week / 1-month returns. NSE's as-traded prices first; Yahoo
    (with its unreliable split adjustments undone) only if NSE has nothing."""
    out = {k: np.nan for k in ("list_open", "list_high", "list_low", "list_close", "ret_1w_pct", "ret_1m_pct")}
    if pd.isna(listing_date) or not issue_price or np.isnan(issue_price):
        return {**out, "note": "no listing date or issue price", "price_source": None}
    ld = pd.Timestamp(listing_date).normalize()
    h, source = pd.DataFrame(), None
    if nse is not None:
        try:
            h, source = nse.history(symbol, ld - pd.Timedelta(days=20), ld + pd.Timedelta(days=60)), "nse"
        except DataError:
            h = pd.DataFrame()
    if h.empty:
        h, source = unadjusted_history(symbol, ld - pd.Timedelta(days=20), days=80), "yahoo"
    out["price_source"] = source
    if h.empty:
        return {**out, "note": "no price history"}
    if (h.index < ld - pd.Timedelta(days=1)).sum() >= 3:
        return {**out, "note": "traded before listing date: FPO or re-listing, not an IPO"}
    h = h[h.index >= ld]
    if h.empty:
        return {**out, "note": "no price history on/after listing"}
    if h.index[0] > ld + pd.Timedelta(days=4):
        return {**out, "note": f"history starts {h.index[0].date()}, after listing"}
    first = h.iloc[0]
    out.update(list_open=first["Open"], list_high=first["High"], list_low=first["Low"], list_close=first["Close"])
    if len(h) > 5:
        out["ret_1w_pct"] = (h["Close"].iloc[5] / issue_price - 1) * 100
    if len(h) > 21:
        out["ret_1m_pct"] = (h["Close"].iloc[21] / issue_price - 1) * 100
    return {**out, "note": ""}


def index_closes(start: date, end: date) -> pd.DataFrame:
    """Daily closes of Nifty 50 and India VIX: columns nifty, vix."""
    yf = _yf()
    frames = {}
    for name, tkr in (("nifty", "^NSEI"), ("vix", "^INDIAVIX")):
        h = yf.Ticker(tkr).history(start=start, end=end + timedelta(days=1), auto_adjust=False)
        if h.empty:
            raise DataError(f"Yahoo returned no data for {tkr}")
        h.index = h.index.tz_localize(None).normalize()
        frames[name] = h["Close"]
    return pd.DataFrame(frames).sort_index()
