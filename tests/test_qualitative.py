"""Offline tests for src/qualitative.py and RHP cache invalidation. No network, no Claude calls."""
import json

import pytest

from src import extract_rhp as X
from src import qualitative as Q


def fake_pages() -> list[str]:
    pages = [f"filler page {i} with some text" for i in range(60)]
    pages[3] = "TABLE OF CONTENTS " + "...... " * 10
    pages[10] = "RISK FACTORS\nSome risk text"
    pages[30] = "OUR BUSINESS\nWe make things"
    pages[40] = "INDUSTRY OVERVIEW\nA big industry"
    pages[50] = "OUR PROMOTERS AND PROMOTER GROUP\nPromoter bio"
    pages[55] = "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS\nCases"
    return pages


def test_select_qual_pages_finds_each_section():
    idx = Q.select_qual_pages(fake_pages())
    assert {10, 30, 40, 50, 55} <= set(idx)
    assert idx == sorted(idx) and len(idx) <= Q.MAX_QUAL_PAGES
    assert 3 not in idx                                  # table of contents is skipped


def test_select_qual_pages_empty_when_no_headings():
    assert Q.select_qual_pages(["nothing useful"] * 40) == []


def test_qual_schema_requires_pages_on_points():
    fields = Q.Point.model_json_schema()["properties"]
    assert "page" in fields and "text" in fields
    for model in (Q.Risk, Q.Segment, Q.Concentration, Q.Promoter, Q.LitigationItem):
        assert "page" in model.model_json_schema()["properties"]


def test_cached_qual_is_dropped_when_rhp_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(X, "RHP_DIR", tmp_path / "rhp")
    X.RAW_DIR.mkdir()
    X.RHP_DIR.mkdir()
    pdf = X.RHP_DIR / "ABC.pdf"
    pdf.write_bytes(b"%PDF-1.4 first version")
    rec = {"symbol": "ABC", "rhp_sha256": X.file_sha256(pdf), "qual": {"risks": []}}
    Q.qual_path("ABC").write_text(json.dumps(rec), encoding="utf-8")
    assert Q.extract_qual("ABC")["qual"]["risks"] == []        # same file: served from cache
    pdf.write_bytes(b"%PDF-1.4 a different RHP")
    with pytest.raises(Exception):                              # stale cache removed; the fake PDF cannot be parsed, so the fresh run fails loudly
        Q.extract_qual("ABC", client=object())
    assert not Q.qual_path("ABC").exists()


def test_clear_cached_removes_all_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(X, "RHP_DIR", tmp_path / "rhp")
    X.RAW_DIR.mkdir()
    X.RHP_DIR.mkdir()
    for p in (X.RAW_DIR / "ABC.json", X.RAW_DIR / "ABC.qual.json", X.RHP_DIR / "ABC.pages.json"):
        p.write_text("{}")
    X.clear_cached("ABC")
    assert not list(X.RAW_DIR.iterdir()) and not list(X.RHP_DIR.iterdir())


# ---- clean_qual: one segment split, no repeats ---------------------------- #
def seg(name, dimension, period, share=None, page=1):
    return {"name": name, "dimension": dimension, "period": period, "share_of_revenue": share, "page": page}


def test_clean_qual_keeps_one_segment_split_for_the_latest_year():
    q = {"segments": [
        seg("Raw coal", "by product", "Fiscal 2025"), seg("Washed coal", "by product", "Fiscal 2025"),
        seg("Raw coal", "by product", "Fiscal 2026"), seg("Washed coal", "by product", "Fiscal 2026"), seg("Rejects", "by product", "Fiscal 2026"),
        seg("Power", "by customer", "Fiscal 2026"), seg("Cement", "by customer", "Fiscal 2026"),
        seg("Raw coal", "by product", "Six months ended September 30, 2026")]}
    out = Q.clean_qual(q)["segments"]
    assert [s["name"] for s in out] == ["Raw coal", "Washed coal", "Rejects"]
    assert {s["period"] for s in out} == {"Fiscal 2026"} and {s["dimension"] for s in out} == {"by product"}


def test_clean_qual_dedupes_names_and_points():
    q = {"segments": [seg("Retail", "by channel", "FY2025"), seg("retail ", "by channel", "FY2025"), seg("Online", "by channel", "FY2025")],
         "business_summary": [{"text": "It is the largest coal producer in India.", "page": 3}, {"text": "It is the largest coal producer in India", "page": 9},
                              {"text": "It has 12 mines.", "page": 4}],
         "risks": [{"title": "Customer concentration", "detail": "a", "severity": "high", "page": 5},
                   {"title": "customer concentration", "detail": "b", "severity": "high", "page": 6}],
         "promoters": [{"name": "Coal India Limited", "background": "x", "page": 1}, {"name": "Coal India Limited", "background": "y", "page": 2}]}
    out = Q.clean_qual(q)
    assert [s["name"] for s in out["segments"]] == ["Retail", "Online"]
    assert [p["text"] for p in out["business_summary"]] == ["It is the largest coal producer in India.", "It has 12 mines."]
    assert len(out["risks"]) == 1 and len(out["promoters"]) == 1


def test_clean_qual_handles_old_caches_with_period_in_the_name():
    q = {"segments": [{"name": "Raw coal (six months ended September 30, 2026)", "page": 1}, {"name": "Raw coal (Fiscal 2026)", "page": 2},
                      {"name": "Washed coal (Fiscal 2026)", "page": 2}, {"name": "Washed coal (six months ended September 30, 2026)", "page": 1}]}
    out = Q.clean_qual(q)["segments"]
    assert [s["name"] for s in out] == ["Raw coal", "Washed coal"] and all("Fiscal 2026" in s["period"] for s in out)


def test_clean_qual_is_safe_on_empty_input():
    out = Q.clean_qual({})
    assert out["segments"] == [] and out["risks"] == [] and out["business_summary"] == []


def test_segment_schema_asks_for_dimension_and_period():
    props = Q.Segment.model_json_schema()["properties"]
    assert "dimension" in props and "period" in props
    assert "latest full financial year" in Q.QualExtract.model_json_schema()["properties"]["segments"]["description"]
