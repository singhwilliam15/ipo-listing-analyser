"""Live subscription snapshots for every open issue, appended to data/snapshots.csv.

    python -m src.build snapshot            # one snapshot of every ACTIVE issue
    python -m src.build snapshot --all      # also issues that have not opened yet

Safe to run from cron every 30 minutes during market hours:
- an exclusive lock file stops two runs overlapping; a run that cannot get the lock exits quietly,
- a snapshot is written only when NSE's own "updated as on" stamp changed for that symbol, so a cron job that
  runs more often than NSE refreshes adds no duplicate rows,
- NSE errors are logged and never raise, so one bad issue or a blocked request leaves the file intact,
- every write appends whole rows in one call, after the header if the file is new.

Example crontab (IST machine, Mon-Fri 09:30-17:30, every 30 minutes):
    */30 9-17 * * 1-5  cd /path/to/ipo-analyser && .venv/bin/python -m src.build snapshot
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .fetch_market import DataError, NSEClient, parse_categories

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "snapshots.csv"
LOCK_STALE_SECONDS = 20 * 60

COLUMNS = ["captured_at_utc", "nse_updated", "symbol", "company", "segment", "status", "price_low", "price_high",
           "issue_open", "issue_close",
           "sub_total_x", "sub_qib_x", "sub_nii_x", "sub_bnii_x", "sub_snii_x", "sub_retail_x", "sub_employee_x", "sub_shareholder_x",
           "shares_total", "shares_qib", "shares_nii", "shares_retail", "shares_employee", "shares_shareholder"]


class _Lock:
    """Exclusive lock via O_EXCL file creation (works on Windows, macOS and Linux). A lock older than
    LOCK_STALE_SECONDS is taken to be left over from a crashed run and is replaced."""

    def __init__(self, path: Path):
        self.path, self.fd = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
                        self.path.unlink()
                        continue
                except OSError:
                    pass
                return None
        return None

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                self.path.unlink()
            except OSError:
                pass


def _last_stamps(path: Path) -> dict[str, str]:
    """symbol -> NSE 'updated as on' stamp of the newest stored row."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["symbol"]] = row.get("nse_updated", "")
    return out


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    if new:
        w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in COLUMNS})
    with open(path, "a", newline="", encoding="utf-8") as f:       # one write call per run
        f.write(buf.getvalue())


def take_snapshot(nse: NSEClient | None = None, path: Path = SNAPSHOTS, include_upcoming: bool = False,
                  now: datetime | None = None, log=print) -> int:
    """Append one row per open issue whose NSE figures changed. Returns the number of rows written."""
    from .live import list_open_issues          # imported here: live pulls in the modelling stack
    with _Lock(path.with_suffix(".lock")) as lock:
        if lock is None:
            log("snapshot: another run holds the lock, skipping")
            return 0
        try:
            nse = nse or NSEClient()
            issues = list_open_issues(nse)
        except Exception as exc:                # NSE blocked, offline, or the format changed: log, never raise
            log(f"snapshot: could not list issues ({exc})")
            return 0
        if not len(issues):
            return 0
        if not include_upcoming:
            issues = issues[issues["status"].astype(str).str.lower() == "active"]
        seen = _last_stamps(path)
        stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for r in issues.to_dict("records"):
            try:
                cats = parse_categories(nse.ipo_detail(r["symbol"]))
            except Exception as exc:
                log(f"snapshot: {r['symbol']} skipped ({exc})")
                continue
            if cats.get("sub_total_x") is None and cats.get("sub_qib_x") is None:
                continue                         # no bids published yet
            if cats.get("nse_updated") and seen.get(r["symbol"]) == cats["nse_updated"]:
                continue                         # NSE has not refreshed since our last row
            rows.append({"captured_at_utc": stamp, "symbol": r["symbol"], "company": r.get("company"), "segment": r.get("segment"),
                         "status": r.get("status"), "price_low": r.get("price_low"), "price_high": r.get("price_high"),
                         "issue_open": r.get("issue_open"), "issue_close": r.get("issue_close"), **cats})
        _append(path, rows)
        log(f"snapshot: {len(rows)} new row(s) for {len(issues)} issue(s)")
        return len(rows)
