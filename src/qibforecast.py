"""Forecast of an IPO's final QIB subscription before bidding, as a mean and a spread in log1p space.

The pre-bid estimate in report.py does not guess a probability directly. It first asks a forecaster "where will QIB
subscription end up?", spreads that into a distribution, and pushes 21 points of it through the demand model and
Model A. Any object with the interface of `QibForecaster` can be plugged in via `Models.forecaster`.

    mean, sd = forecaster.forecast(rows)                       # arrays, one entry per row
    mean, sd = forecaster.forecast(rows, snapshots={...})      # once a conditioned forecaster exists (see below)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "l_qib"                    # log1p(final QIB subscription)
RIDGE_ALPHAS = (1.0, 10.0, 30.0, 100.0, 300.0)
CV_FOLDS = 5


class QibForecaster:
    """Interface. `fit` sees training rows only; `forecast` returns (mean, sd) of log1p(final QIB) per row."""

    def fit(self, train: pd.DataFrame, features: list[str]) -> "QibForecaster":
        raise NotImplementedError

    def forecast(self, rows: pd.DataFrame, snapshots: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class RidgeQibForecaster(QibForecaster):
    """Ridge regression of log1p(final QIB) on the pre-bid features (fitted inside the training rows only).

    The spread is the residual SD of 5-fold out-of-fold predictions, so it is not flattered by in-sample fit."""

    def __init__(self, alphas=RIDGE_ALPHAS, cv: int = CV_FOLDS):
        self.alphas, self.cv = alphas, cv
        self.features: list[str] = []
        self.pipe = None
        self.sd_: float = 1.0
        self.n_: int = 0
        self.r2_oof_: float = float("nan")

    def fit(self, train: pd.DataFrame, features: list[str]) -> "RidgeQibForecaster":
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold, cross_val_predict

        from .report import _num_pipe          # imported here: report imports this module
        rows = train[train[TARGET].notna()]
        self.features, self.n_ = list(features), len(rows)
        X, y = rows[self.features], rows[TARGET].to_numpy()
        self.pipe = _num_pipe(RidgeCV(alphas=self.alphas)).fit(X, y)
        oof = cross_val_predict(_num_pipe(RidgeCV(alphas=self.alphas)), X, y, cv=KFold(self.cv, shuffle=True, random_state=0))
        resid = y - oof
        self.sd_ = float(np.std(resid, ddof=1))
        self.r2_oof_ = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
        return self

    def _x(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows.copy()
        for c in self.features:
            if c not in X:
                X[c] = np.nan
        return X[self.features].apply(pd.to_numeric, errors="coerce")

    def forecast(self, rows: pd.DataFrame, snapshots: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(self.pipe.predict(self._x(rows)), dtype=float)
        return mu, np.full(len(mu), self.sd_)


class SnapshotQibForecaster(RidgeQibForecaster):
    """STUB: the fundamentals forecast, to be conditioned on early-bidding QIB from data/snapshots.csv.

    TODO (not implemented, the data does not exist yet). `python -m src.build snapshot` has been logging NSE's
    category subscription with timestamps only since it was added, so there are no historical Day-1 / Day-2 QIB figures
    to learn from. Once the log covers enough closed issues (roughly 30 or more mainboard IPOs, a few months of cron):

    1. Build a history table from data/snapshots.csv: for each symbol, the last snapshot of bidding day 1 and day 2
       (use `issue_open` and the NSE `nse_updated` stamp to assign a day index), with sub_qib_x and sub_total_x, joined to
       the final QIB subscription in data/ipos.csv.
    2. Fit, on training issues only, a ridge of the final log1p(QIB) on [the fundamentals forecast mean from this class,
       log1p(day-k QIB), log1p(day-k total), day index k]. QIB bids arrive mostly on the last day, so the weight on the
       fundamentals forecast should fall only slowly; let the data say how fast.
    3. Return the conditioned mean and a smaller residual SD (out-of-fold, per day index) from `forecast` when `snapshots`
       carries {"day": k, "sub_qib_x": ..., "sub_total_x": ...}; keep falling back to the plain ridge forecast when it is None.
    4. Add it to the walk-forward in report.py with a day-1 and day-2 variant and compare against this class before using it.
    """

    def forecast(self, rows: pd.DataFrame, snapshots: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        # TODO: condition on `snapshots` as described above. Until then the snapshots are ignored.
        return super().forecast(rows, None)
