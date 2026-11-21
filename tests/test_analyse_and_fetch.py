import numpy as np
import pandas as pd
import pytest

from src import analyse, demo, schema
from src.fetch_market import find_rhp_url, parse_bids, parse_past_issues, price_band


# --------------------------------------------------------------------------- #
# analyse
# --------------------------------------------------------------------------- #
def test_metrics():
    m = analyse.metrics([1, 1, 0, 0], [1, 0, 1, 0], gains=[10, 5, -3, -8])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["avg_gain_if_applied_pct"] == pytest.approx(3.5)


def test_baselines():
    df = pd.DataFrame({"sub_total_x": [50, 2, np.nan]})
    assert analyse.baseline_always(df).tolist() == [1, 1, 1]
    assert analyse.baseline_sub10(df).tolist() == [1, 0, 0]


@pytest.fixture(scope="module")
def synth():
    return analyse.add_split(demo.synthetic(400))


def test_demo_matches_schema(synth):
    assert list(synth.columns[:-1]) == schema.ORDER
    assert set(synth["split"]) == {"train", "test"}


def test_test_rows_never_used_for_fitting(synth):
    """Scrambling every test outcome must not change the chosen rules or the model."""
    res_a = analyse.evaluate_all(synth)
    b = synth.copy()
    t = b["split"] == "test"
    b.loc[t, "target_gain"] = 1 - b.loc[t, "target_gain"]
    res_b = analyse.evaluate_all(b)
    assert str(res_a["rules"]) == str(res_b["rules"])
    pd.testing.assert_frame_equal(res_a["coefficients"], res_b["coefficients"])


def test_pipeline_finds_planted_signal(synth):
    res = analyse.evaluate_all(synth)
    top = res["groups"]["feature"].head(8).tolist()
    assert any(f.startswith("sub_") for f in top)        # demo outcome is driven by subscription
    test = res["test_scores"]
    assert test.loc["Logistic (p > 0.5)", "accuracy"] > test.loc["Always apply", "accuracy"] - 0.05
    md = analyse.report_markdown(res)
    assert "Base rate" in md and "Test scores" in md


def test_too_few_rows_reports_error(synth):
    res = analyse.evaluate_all(synth.head(10))
    assert "error" in res


# --------------------------------------------------------------------------- #
# NSE parsers
# --------------------------------------------------------------------------- #
def test_price_band():
    assert price_band("Rs.285 to Rs.300") == (285, 300)
    assert price_band("1,140") == (1140, 1140)
    assert all(np.isnan(price_band(None)))


def test_parse_past_issues():
    rows = [{"symbol": "abc", "companyName": "ABC Ltd", "securityType": "EQ", "ipoStartDate": "02-DEC-2024",
             "ipoEndDate": "04-DEC-2024", "listingDate": "09-DEC-2024", "priceRange": "Rs.95 to Rs.100", "issuePrice": "100"},
            {"symbol": "XYZ", "companyName": "XYZ Ltd", "securityType": "SME", "listingDate": "10-01-2025", "issuePrice": "50"},
            {"symbol": "NCD1", "companyName": "Bond Co", "securityType": "DEBT", "issuePrice": "######"},
            {"symbol": "RT", "companyName": "A REIT", "securityType": "REIT", "issuePrice": "100"},
            {"symbol": "ADANIENPP1", "companyName": "Adani Enterprises", "securityType": "Equity", "issuePrice": "450"}]
    df = parse_past_issues(rows).set_index("symbol")
    assert set(df.index) == {"ABC", "XYZ"}
    assert df.at["ABC", "segment"] == "Mainboard" and df.at["XYZ", "segment"] == "SME"
    assert df.at["ABC", "listing_date"] == pd.Timestamp("2024-12-09")
    assert (df.at["ABC", "price_low"], df.at["ABC", "issue_price"]) == (95, 100)
    assert df.at["XYZ", "listing_date"] == pd.Timestamp("2025-01-10")


def test_parse_bids():
    detail = {"bidDetails": {"data": [
        {"category": "Qualified Institutional Buyers (QIBs)", "noOfTimes": "120.5"},
        {"category": "Anchor Investors", "noOfTimes": "1"},
        {"category": "Non Institutional Investors", "noOfTimes": "80.1"},
        {"category": "bNII (bids above Rs 10 lakhs)", "noOfTimes": "90"},
        {"category": "sNII (bids below Rs 10 lakhs)", "noOfTimes": "60"},
        {"category": "Retail Individual Investors (RIIs)", "noOfTimes": "12.3"},
        {"category": "Employees", "noOfTimes": "3"},
        {"category": "Total", "noOfTimes": "55.2", "noOfApplications": "1,234,567"},
    ]}}
    b = parse_bids(detail)
    assert b == {"sub_qib_x": 120.5, "sub_nii_x": 80.1, "sub_bnii_x": 90, "sub_snii_x": 60, "sub_retail_x": 12.3,
                 "sub_employee_x": 3, "sub_total_x": 55.2, "n_applications": 1234567}


def test_find_rhp_url():
    detail = {"docs": [{"title": "Addendum", "link": "/a/add.pdf"}, {"title": "Red Herring Prospectus", "link": "/a/rhp.pdf"}]}
    assert find_rhp_url(detail) == "https://www.nseindia.com/a/rhp.pdf"
    assert find_rhp_url({"x": 1}) is None


def test_real_nse_payload_shape():
    """Shape copied from NSE ipo-detail for SURAKSHA (Dec 2024)."""
    from src.fetch_market import issue_info
    detail = {
        "bidDetails": [{"category": "Total", "noOfTime": "0.49"}],          # NSE-only: must be ignored
        "activeCat": {"dataList": [
            {"category": "Category", "noOfTotalMeant": "No. of times of total meant for the category"},
            {"category": "Qualified Institutional Buyers(QIBs)", "noOfTotalMeant": "1.737"},
            {"category": "Non Institutional Investors", "noOfTotalMeant": "1.405"},
            {"category": "Non Institutional Investors(Bid amount of more than Ten Lakh Rupees)", "noOfTotalMeant": "1.527"},
            {"category": "Individuals(Other than RIIs)", "noOfTotalMeant": ""},
            {"category": "Non Institutional Investors(Bid amount of more than Two Lakh Rupees upto Ten Lakh Rupees)", "noOfTotalMeant": "1.162"},
            {"category": "Retail Individual Investors(RIIs)", "noOfTotalMeant": "0.946"},
            {"category": "Total", "noOfTotalMeant": "1.270"}]},
        "issueInfo": {"dataList": [
            {"title": "Issue Size", "value": "\"Initial Public Offer of Offer for Sale Up to 19,189,330 Equity Shares (including anchor investor portion of 57,56,797 Equity Shares)\""},
            {"title": "Bid Lot", "value": "34 Equity Shares and in multiples thereof"},
            {"title": "Book Running Lead Managers", "value": "\"ICICI Securities Limited, Nuvama Wealth Management Limited and SBI Capital Markets Limited\""},
            {"title": "Red Herring Prospectus", "value": "https://nsearchives.nseindia.com/content/ipo/RHP_SURAKSHA.zip"}]},
    }
    b = parse_bids(detail)
    assert (b["sub_qib_x"], b["sub_nii_x"], b["sub_bnii_x"], b["sub_snii_x"], b["sub_retail_x"], b["sub_total_x"]) == \
        (1.737, 1.405, 1.527, 1.162, 0.946, 1.270)
    info = issue_info(detail)
    assert info["lot_size_nse"] == 34
    assert info["lead_managers_nse"] == "ICICI Securities Limited; Nuvama Wealth Management Limited; SBI Capital Markets Limited"
    assert info["anchor_shares_cr"] == pytest.approx(0.5756797)
    assert find_rhp_url(detail).endswith("RHP_SURAKSHA.zip")


def test_pick_rhp_from_zip():
    import io
    import zipfile
    from src.fetch_market import pick_rhp_from_zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X/Co - NSE Letter for Corrigendum.pdf", b"%PDF corr" * 1000)
        z.writestr("X/Co- Red Herring Prospectus 25-11-2024.pdf", b"%PDF rhp")
        z.writestr("X/Co - GID - Final version.pdf", b"%PDF gid" * 500)
    assert pick_rhp_from_zip(buf.getvalue()) == b"%PDF rhp"


def test_parse_nse_history_prefers_eq_series():
    from src.fetch_market import parse_nse_history
    rows = [{"CH_SERIES": "BE", "mTIMESTAMP": "10-Jul-2024", "CH_OPENING_PRICE": 1, "CH_TRADE_HIGH_PRICE": 1, "CH_TRADE_LOW_PRICE": 1, "CH_CLOSING_PRICE": 1},
            {"CH_SERIES": "EQ", "mTIMESTAMP": "10-Jul-2024", "CH_OPENING_PRICE": 356, "CH_TRADE_HIGH_PRICE": 370, "CH_TRADE_LOW_PRICE": 340, "CH_CLOSING_PRICE": 350.35},
            {"CH_SERIES": "EQ", "mTIMESTAMP": "11-Jul-2024", "CH_OPENING_PRICE": 353.4, "CH_TRADE_HIGH_PRICE": 365, "CH_TRADE_LOW_PRICE": 350, "CH_CLOSING_PRICE": 362.6}]
    h = parse_nse_history(rows)
    assert list(h.index) == [pd.Timestamp("2024-07-10"), pd.Timestamp("2024-07-11")]
    assert h.iloc[0]["Open"] == 356 and h.iloc[0]["Close"] == 350.35


def test_zip_without_rhp_is_rejected():
    import io
    import zipfile
    from src.fetch_market import DataError, pick_rhp_from_zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X/RHP_Corrigendum.pdf", b"%PDF c")
        z.writestr("X/Uniparts_GID.pdf", b"%PDF g" * 100)
    with pytest.raises(DataError):
        pick_rhp_from_zip(buf.getvalue())
