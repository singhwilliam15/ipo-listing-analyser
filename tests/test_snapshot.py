"""Offline tests for the subscription snapshot logger and the category parser. No network."""
import csv
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import fetch_market as FM
from src import live, snapshot


def payload(updated="20-Nov-2026 17:03:00", qib=1.37, total=0.48, employee=0.09, shareholder=0.28):
    row = lambda sr, cat, off, times: {"srNo": sr, "category": cat, "noOfShareOffered": off, "noOfTotalMeant": times}
    return {"activeCat": {"updateTime": f"Updated as on {updated}", "dataList": [
        row("Sr.No.", "Category", "No.of shares offered/reserved", "No. of times of total meant for the category"),
        row("1", "Qualified Institutional Buyers(QIBs)", "1000", str(qib)),
        row("1(a)", "Foreign Institutional Investors(FIIs)", "", ""),
        row("2", "Non Institutional Investors", "800", "0.18"),
        row("2.1", "Non Institutional Investors(Bid amount of more than Ten Lakh Rupees)", "500", "0.13"),
        row("2.2", "Non Institutional Investors(Bid amount of more than Two Lakh Rupees upto Ten Lakh Rupees)", "300", "0.27"),
        row("3", "Retail Individual Investors(RIIs)", "1900", "0.24"),
        row("4", "Employees", "300", str(employee)),
        row("5", "Shareholders", "600", str(shareholder)),
        row(None, "Total", "4.5E3", str(total))]}}


def test_parse_categories_keeps_employee_and_shareholder_quotas():
    c = FM.parse_categories(payload())
    assert c["sub_qib_x"] == pytest.approx(1.37) and c["sub_total_x"] == pytest.approx(0.48)
    assert c["sub_nii_x"] == pytest.approx(0.18) and c["sub_bnii_x"] == pytest.approx(0.13) and c["sub_snii_x"] == pytest.approx(0.27)
    assert c["sub_retail_x"] == pytest.approx(0.24)
    assert c["sub_employee_x"] == pytest.approx(0.09) and c["sub_shareholder_x"] == pytest.approx(0.28)
    assert c["shares_shareholder"] == 600 and c["shares_total"] == 4500
    assert c["nse_updated"] == "20-Nov-2026 17:03:00"


def test_parse_categories_handles_missing_blocks():
    c = FM.parse_categories({})
    assert c["sub_qib_x"] is None and c["nse_updated"] is None
    c = FM.parse_categories({"activeCat": {"dataList": [{"srNo": "5", "category": "Shareholders", "noOfShareOffered": "", "noOfTotalMeant": ""}]}})
    assert c["sub_shareholder_x"] is None                         # empty cells are not zeros


class FakeNSE:
    def __init__(self, details, issues=None):
        self.details, self.calls = details, 0
        self.issues = issues if issues is not None else [
            {"symbol": "AAA", "company": "A Ltd", "status": "Active", "segment": "Mainboard", "price_low": 90.0, "price_high": 100.0,
             "issue_open": "19-Nov-2026", "issue_close": "23-Nov-2026"},
            {"symbol": "BBB", "company": "B Ltd", "status": "Forthcoming", "segment": "Mainboard", "price_low": 10.0, "price_high": 12.0,
             "issue_open": "25-Nov-2026", "issue_close": "27-Nov-2026"}]

    def ipo_detail(self, symbol):
        self.calls += 1
        d = self.details[symbol]
        if isinstance(d, Exception):
            raise d
        return d


@pytest.fixture
def stub_issues(monkeypatch):
    def install(nse):
        monkeypatch.setattr(live, "list_open_issues", lambda n=None: pd.DataFrame(nse.issues))
    return install


NOW = datetime(2026, 11, 21, 4, 30, tzinfo=timezone.utc)


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_snapshot_writes_header_and_all_categories(tmp_path, stub_issues):
    nse = FakeNSE({"AAA": payload(), "BBB": payload()})
    stub_issues(nse)
    path = tmp_path / "snap.csv"
    assert snapshot.take_snapshot(nse, path, now=NOW, log=lambda m: None) == 1       # BBB has not opened
    r = rows(path)
    assert [x["symbol"] for x in r] == ["AAA"] and r[0]["captured_at_utc"] == "2026-11-21T04:30:00+00:00"
    assert float(r[0]["sub_employee_x"]) == pytest.approx(0.09) and float(r[0]["sub_shareholder_x"]) == pytest.approx(0.28)
    assert r[0]["nse_updated"] == "20-Nov-2026 17:03:00"
    assert list(r[0].keys()) == snapshot.COLUMNS


def test_snapshot_includes_upcoming_on_request(tmp_path, stub_issues):
    nse = FakeNSE({"AAA": payload(), "BBB": payload()})
    stub_issues(nse)
    assert snapshot.take_snapshot(nse, tmp_path / "s.csv", include_upcoming=True, now=NOW, log=lambda m: None) == 2


def test_snapshot_is_idempotent_until_nse_refreshes(tmp_path, stub_issues):
    nse = FakeNSE({"AAA": payload(), "BBB": payload()})
    stub_issues(nse)
    path = tmp_path / "snap.csv"
    log = lambda m: None
    assert snapshot.take_snapshot(nse, path, now=NOW, log=log) == 1
    assert snapshot.take_snapshot(nse, path, now=NOW, log=log) == 0          # same NSE stamp: cron ran again, nothing new
    assert len(rows(path)) == 1
    nse.details["AAA"] = payload(updated="21-Nov-2026 10:30:00", qib=3.0)
    assert snapshot.take_snapshot(nse, path, now=NOW, log=log) == 1
    r = rows(path)
    assert len(r) == 2 and float(r[1]["sub_qib_x"]) == pytest.approx(3.0)
    assert open(path, encoding="utf-8").read().count("captured_at_utc") == 1  # header written once


def test_snapshot_survives_nse_errors(tmp_path, stub_issues, monkeypatch):
    path = tmp_path / "snap.csv"
    nse = FakeNSE({"AAA": FM.DataError("HTTP 403"), "BBB": payload()})
    stub_issues(nse)
    assert snapshot.take_snapshot(nse, path, now=NOW, log=lambda m: None) == 0      # one issue failing is skipped, not fatal
    assert not path.exists()

    def boom(n=None):
        raise FM.DataError("NSE unreachable")
    monkeypatch.setattr(live, "list_open_issues", boom)
    assert snapshot.take_snapshot(nse, path, now=NOW, log=lambda m: None) == 0      # listing failure is swallowed too
    assert not path.exists()


def test_snapshot_skips_issues_with_no_bids_yet(tmp_path, stub_issues):
    empty = {"activeCat": {"updateTime": "Updated as on 20-Nov-2026 09:00:00", "dataList": []}}
    nse = FakeNSE({"AAA": empty, "BBB": empty})
    stub_issues(nse)
    assert snapshot.take_snapshot(nse, tmp_path / "s.csv", now=NOW, log=lambda m: None) == 0


def test_snapshot_lock_prevents_overlapping_runs(tmp_path, stub_issues):
    nse = FakeNSE({"AAA": payload(), "BBB": payload()})
    stub_issues(nse)
    path = tmp_path / "snap.csv"
    lock = path.with_suffix(".lock")
    lock.write_text("123")                                                  # another run is in progress
    msgs = []
    assert snapshot.take_snapshot(nse, path, now=NOW, log=msgs.append) == 0
    assert nse.calls == 0 and not path.exists() and any("lock" in m for m in msgs)
    import os, time
    old = time.time() - 2 * snapshot.LOCK_STALE_SECONDS                      # a crashed run's stale lock is replaced
    os.utime(lock, (old, old))
    assert snapshot.take_snapshot(nse, path, now=NOW, log=lambda m: None) == 1
    assert not lock.exists()                                                # lock released afterwards


def test_snapshot_cli_is_registered():
    from src import build
    import inspect
    assert "snapshot" in inspect.getsource(build.main) and hasattr(build, "cmd_snapshot")
