"""Forecast models with a common interface: fit on a history, forecast h steps given future exog."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def naive(y: pd.Series, h: int, **_: object) -> np.ndarray:
    return np.repeat(y.iloc[-1], h)


def ets(y: pd.Series, h: int, **_: object) -> np.ndarray:
    fit = ExponentialSmoothing(y.values, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return np.asarray(fit.forecast(h))


def arimax(y: pd.Series, h: int, x: pd.Series | None = None, x_future: np.ndarray | None = None,
           order: tuple[int, int, int] = (1, 1, 0), return_fit: bool = False):  # type: ignore[no-untyped-def]
    exog = None if x is None else x.values.reshape(-1, 1)
    fit = SARIMAX(y.values, exog=exog, order=order, trend="n").fit(disp=False)
    fx = None if x_future is None else np.asarray(x_future).reshape(-1, 1)
    fc = fit.get_forecast(h, exog=fx)
    if return_fit:
        return fit, fc
    return np.asarray(fc.predicted_mean)


def rolling_origin(y: pd.Series, x: pd.Series | None, first_origin: str, h: int = 4,
                   min_train: int = 12) -> pd.DataFrame:
    """Expanding-window backtest. ARIMAX gets the realised exog path (conditional forecast)."""
    rows = []
    origins = [i for i, d in enumerate(y.index) if d >= pd.Timestamp(first_origin) and i >= min_train - 1]
    for i in origins:
        train = y.iloc[: i + 1]
        test = y.iloc[i + 1: i + 1 + h]
        if len(test) == 0:
            continue
        hh = len(test)
        preds = {"Naive": naive(train, hh), "ETS (damped)": ets(train, hh)}
        if x is not None:
            preds["ARIMAX(1,1,0)+SAIBOR"] = arimax(train, hh, x.iloc[: i + 1], x.iloc[i + 1: i + 1 + hh].values)
        for name, p in preds.items():
            for k in range(hh):
                rows.append(dict(origin=y.index[i], h=k + 1, model=name, actual=test.iloc[k], pred=p[k]))
    out = pd.DataFrame(rows)
    out["err"] = out.pred - out.actual
    return out


def score(bt: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """RMSE / MAE in the target's units x scale (for log targets: ~percent)."""
    g = bt.groupby(["model", "h"]).err
    s = pd.DataFrame({"RMSE": g.apply(lambda e: scale * np.sqrt(np.mean(e ** 2))), "MAE": g.apply(lambda e: scale * np.mean(np.abs(e))),
                      "n": g.size()}).reset_index()
    return s
