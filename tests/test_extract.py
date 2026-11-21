import json
from types import SimpleNamespace

import pytest

from src import extract_rhp as ex


def fig(v=None, unit=None, page=None):
    return ex.Fig(value=v, unit=unit, page=page)


def blank(**kw) -> ex.RHPExtract:
    names = [n for n, f in ex.RHPExtract.model_fields.items() if f.annotation is ex.Fig]
    base = {n: fig() for n in names}
    base.update(company="Test Ltd", sector="IT & software", exchanges="both", selling_shareholders=[],
                objects_of_offer=[], pre_ipo_placement=None, lead_managers=[], financials_basis="consolidated",
                fy0_label="FY2024", stub_period_months=None, no_listed_peers=False, peers=[], notes=[])
    base.update(kw)
    return ex.RHPExtract(**base)


def test_units_convert_to_crore():
    assert ex.crore(fig(1234.5, "million")) == pytest.approx(123.45)
    assert ex.crore(fig(5000, "lakh")) == pytest.approx(50)
    assert ex.crore(fig(2, "billion")) == pytest.approx(200)
    assert ex.crore(fig(75, "crore")) == 75
    assert ex.crore(fig(None, "million")) is None


def test_flatten_with_pages_and_helpers():
    x = blank(revenue_fy0=fig(2000, "million", 310), pat_fy0=fig(150, "million", 311),
              ofs_shares=fig(12_000_000, "shares", 9),
              pre_issue_shares=fig(100_000_000, "shares", 80),
              selling_shareholders=[ex.Seller(name="Promoter A", kind="promoter", shares_offered=fig(8_000_000, "shares", 9)),
                                    ex.Seller(name="PE Fund", kind="investor", shares_offered=fig(4_000_000, "shares", 9))],
              objects_of_offer=[ex.Proceed(purpose="Repay loans", category="debt_repayment", amount=fig(500, "million", 95)),
                                ex.Proceed(purpose="GCP", category="gcp", amount=fig(100, "million", 96))],
              lead_managers=["Bank One", "Bank Two"],
              peers=[ex.Peer(name="Peer A", nse_symbol="PEERA", pe=40.0, ronw_pct=15.0, page=120),
                     ex.Peer(name="Peer B", nse_symbol=None, pe=None, ronw_pct=12.0, page=120)])
    f = ex.flatten(x)
    assert f["revenue_fy0"] == pytest.approx(200) and f["rhp_page_revenue_fy0"] == 310
    assert f["ofs_shares_cr"] == pytest.approx(1.2)
    assert f["ofs_promoter_shares_cr"] == pytest.approx(0.8) and f["ofs_investor_shares_cr"] == pytest.approx(0.4)
    assert f["proceeds_debt_repayment_cr"] == pytest.approx(50) and f["proceeds_capex_cr"] == 0
    assert f["lead_managers"] == "Bank One; Bank Two"
    assert f["peers"] == "PEERA; Peer B" and f["peer_pe_list"] == [40.0] and f["n_peers"] == 2
    assert f["rhp_page_peers"] == 120
    assert "rhp_page_cash_fy0" not in f                      # no page for a missing figure
    json.dumps(f)                                            # must be serialisable


def test_checks_catch_unit_mixups():
    f = {"revenue_fy0": 100.0, "revenue_fy1": 1.0, "pat_fy0": 150.0, "retail_quota_pct": 30}
    msgs = " | ".join(ex.checks(f))
    assert "PAT| > revenue" in msgs and ">50x" in msgs and "retail quota" in msgs


def test_select_pages_skips_toc_and_finds_sections():
    pages = ["cover"] * 3
    pages.append("TABLE OF CONTENTS\n" + "\n".join(f"Section {i} ........ {i}" for i in range(20)) + " capital structure objects of the offer")
    pages += ["filler text"] * 10
    pages.append("CAPITAL STRUCTURE shareholding pattern pre-offer post-offer promoter group")
    pages.append("OBJECTS OF THE OFFER net proceeds repayment general corporate purposes")
    pages.append("Restated consolidated statement of profit and loss revenue from operations")
    pages.append("continued table")
    idx = ex.select_pages(pages)
    assert 3 not in idx                                      # TOC page
    assert {14, 15, 16, 17} <= set(idx)                      # sections + the page after the financials
    assert idx == sorted(idx)


def test_extract_end_to_end_with_fake_client(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RAW_DIR", tmp_path / "raw")
    pdf = tmp_path / "ABC.pdf"
    pdf.write_bytes(b"%PDF-fake")
    pages = ["Restated consolidated statement of profit and loss revenue from operations " * 20] * 30
    monkeypatch.setattr(ex, "page_texts", lambda p: pages)

    parsed = blank(revenue_fy0=fig(100, "crore", 5), pat_fy0=fig(10, "crore", 5))
    resp = SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, stop_details=None,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=200), _request_id="req_1")
    client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: resp))
    rec = ex.extract("ABC", pdf=pdf, client=client)
    assert rec["fields"]["revenue_fy0"] == 100 and rec["usage"]["input_tokens"] == 1000
    assert (tmp_path / "raw" / "ABC.json").exists()

    resp.stop_reason = "refusal"
    with pytest.raises(ex.ExtractionError):
        ex.extract("ABC", pdf=pdf, client=client, force=True)
