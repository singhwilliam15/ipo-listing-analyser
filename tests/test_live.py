import numpy as np
import pandas as pd

from src import live


def _hist(n=200, seed=0):
    r = np.random.default_rng(seed)
    sub = np.exp(r.normal(3, 1.5, n))
    gain = np.where(sub > 10, r.normal(25, 15, n), r.normal(-3, 10, n))
    return pd.DataFrame({"sub_total_x": sub, "listing_gain_pct": gain, "loss_making_flag": r.integers(0, 2, n),
                         "debt_to_equity_fy0": r.uniform(0, 3, n), "revenue_cagr_pct": r.uniform(-5, 60, n),
                         "ronw_fy0_pct": r.uniform(-10, 40, n), "pat_margin_fy0_pct": r.uniform(-5, 20, n),
                         "pe_premium_to_peers_pct": r.uniform(-50, 100, n), "proceeds_debt_repayment_pct": r.uniform(0, 100, n),
                         "ipo_premium_to_preipo_pct": r.uniform(-20, 40, n)})


def test_subscription_table_is_monotone_and_complete():
    t = live.subscription_table(_hist())
    assert list(t.index) == live.SUB_LABELS
    assert t["gain_rate"].iloc[-1] > t["gain_rate"].iloc[0]


def test_flags_mark_leverage_and_losses_bad_with_evidence():
    row = pd.Series({"loss_making_flag": 1, "debt_to_equity_fy0": 3.0, "revenue_cagr_pct": 40.0, "ofs_pct": 100.0})
    fl = live.flags(row, _hist())
    kinds = {f.text.split(":")[0]: f.kind for f in fl}
    assert kinds["High leverage"] == "bad"
    assert any(f.kind == "bad" and f.text.startswith("Loss-making") and "similar IPOs" in f.evidence for f in fl)
    assert any(f.kind == "good" and f.text.startswith("Revenue CAGR") for f in fl)
    assert any(f.kind == "info" and "offer for sale" in f.text for f in fl)


def test_flags_ignore_missing_values():
    assert live.flags(pd.Series({"debt_to_equity_fy0": np.nan}), _hist()) == []


def test_find_issue_fuzzy_match(monkeypatch):
    fake = pd.DataFrame([{"symbol": "MAHACOAL", "company": "Mahanadi Coalfields Limited", "status": "Active",
                          "segment": "Mainboard", "price_low": 129.0, "price_high": 136.0,
                          "issue_open": "19-Nov-2026", "issue_close": "23-Nov-2026"}])
    monkeypatch.setattr(live, "list_open_issues", lambda nse=None: fake)
    m = live.find_issue("mahanadi coal", include_past=False)
    assert m and m[0].symbol == "MAHACOAL"
    assert live.find_issue("zzzzzz", include_past=False) == []


def test_date_parsing_handles_nse_and_iso():
    assert live._date("23-Nov-2026") == pd.Timestamp("2026-11-23")
    assert live._date("2026-03-27") == pd.Timestamp("2026-03-27")
    assert pd.isna(live._date(None))


def test_uploaded_rhp_is_saved_as_symbol_pdf(tmp_path, monkeypatch):
    from src import extract_rhp
    monkeypatch.setattr(extract_rhp, "RHP_DIR", tmp_path / "rhp")
    src = tmp_path / "upload.pdf"
    src.write_bytes(b"%PDF-1.4 upload")
    path, label = live.get_document("ABC", None, src, nse=None)
    assert path == tmp_path / "rhp" / "ABC.pdf" and path.read_bytes() == b"%PDF-1.4 upload"
    assert label == "uploaded RHP"
    src.write_bytes(b"%PDF-1.4 newer")                       # a different upload replaces the saved copy
    live.get_document("ABC", None, src, nse=None)
    assert path.read_bytes() == b"%PDF-1.4 newer"


def test_analysis_keeps_record_and_notes_apart(monkeypatch, tmp_path):
    """Regression: record and notes were once swapped by a positional constructor call."""
    from src import extract_rhp
    rec = {"fields": {"sector": "IT & software", "pat_fy0": 10.0, "revenue_fy0": 100.0, "revenue_fy2": 60.0, "pat_fy2": 5.0,
                      "ronw_fy0_pct": 20.0, "debt_to_equity_fy0": 0.3}, "checks": ["a check"]}

    class FakeNSE:
        def ipo_detail(self, symbol):
            return {"metaInfo": {}, "issueInfo": {"dataList": []}}

    monkeypatch.setattr(live, "market_series", lambda: None)
    monkeypatch.setattr(live, "get_document", lambda *a, **k: (tmp_path / "x.pdf", "RHP (test)"))
    monkeypatch.setattr(extract_rhp, "extract", lambda *a, **k: rec)
    monkeypatch.setattr(extract_rhp, "make_client", lambda: None)
    hist = live.load_history()
    m = live.Match("ABC", "Abc Ltd", "Active", "Mainboard", 90.0, 100.0, "01-Dec-2026", "03-Dec-2026")
    a = live.analyse_issue(m, hist=hist, nse=FakeNSE())
    assert a.record is rec
    assert isinstance(a.notes, list) and all(isinstance(n, str) for n in a.notes)
