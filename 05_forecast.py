"""05 - Forecast: Naive, ETS, ARIMAX(+SAIBOR) with rolling-origin validation, then 4-quarter forecasts.

Windows come from shfm.config.windows(): adding a new SAMA/GASTAT release extends everything.

Targets
  A. log real new-mortgage value, seasonally adjusted (quarterly, 2016Q1 -> last complete quarter)
  B. log RPI residential (new method, 2021Q1 -> latest quarter)  -> very short; treat as indicative
Errors are on log scale x100 (~ percent).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shfm.common import PROCESSED, TABLES  # noqa: E402
from shfm.config import MTG_START, RPI_NEW_START, q_end, windows  # noqa: E402
from shfm.models import arimax, ets, naive, rolling_origin, score  # noqa: E402

H = 4


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(PROCESSED / "quarterly_sa.parquet")
    long = pd.read_parquet(PROCESSED / "long.parquet")
    return q, long


def saibor_base_path(long: pd.DataFrame) -> float:
    """Base case: SAIBOR stays at its latest observed level (average of the last 3 available months)."""
    s = long[long.series == "saibor_3m"].dropna(subset=["value"]).sort_values("date").value
    return float(s.tail(3).mean())


def nsa_baselines(q: pd.DataFrame, end: pd.Timestamp, first_origin: pd.Timestamp) -> pd.DataFrame:
    """Backtest of baselines on the NON-seasonally-adjusted real series (what decision-makers see)."""
    q = q.loc[MTG_START:end]
    y, ysa = np.log(q.mtg_real_value_total), np.log(q.mtg_real_sa)
    sf = y - ysa
    rows, idx = [], list(q.index)
    for i, d in enumerate(idx):
        if d < first_origin:
            continue
        for h in range(1, H + 1):
            if i + h >= len(idx):
                break
            t, act = idx[i + h], y.iloc[i + h]
            hist = sf.iloc[: i + 1]
            preds = {"SA-Naive re-seasonalised": ysa.iloc[i] + hist[hist.index.quarter == t.quarter].mean(),
                     "Seasonal naive (same q last year)": y.iloc[i + h - 4],
                     "Seasonal naive + last YoY": y.iloc[i + h - 4] + (y.iloc[i] - y.iloc[i - 4])}
            rows += [dict(model=k, h=h, err=v - act) for k, v in preds.items()]
    b = pd.DataFrame(rows)
    return b.groupby(["model", "h"]).err.apply(lambda e: 100 * np.sqrt((e ** 2).mean())).unstack().round(1)


def main() -> dict[str, object]:
    q, long = load()
    win = windows(long)
    FC_INDEX = win.fc_index
    m = q.loc[MTG_START:win.mtg_end]
    yA, xA = np.log(m.mtg_real_sa), m.saibor_3m
    r = q.loc[RPI_NEW_START:win.rpi_end]
    yB, xB = np.log(r.rpi_residential), r.saibor_3m.shift(1).bfill()

    originA, originB = q_end(win.mtg_end, -14), q_end(win.rpi_end, -9)  # backtest over the last ~3.5 / ~2.25 years
    btA = rolling_origin(yA, xA, first_origin=str(originA.date()), h=H)
    btB = rolling_origin(yB, xB, first_origin=str(originB.date()), h=H, min_train=10)
    sA, sB = score(btA).assign(target="real mortgage value (SA)"), score(btB).assign(target="RPI residential")
    scores = pd.concat([sA, sB])
    scores.round(2).to_csv(TABLES / "backtest_scores.csv", index=False)

    nsa = nsa_baselines(q, win.mtg_end, originA)
    nsa.to_csv(TABLES / "backtest_nsa_baselines.csv")

    s0 = saibor_base_path(long)
    x_future = np.repeat(s0, H)
    fitA, fcA = arimax(yA, H, xA, x_future, return_fit=True)
    ciA = fcA.conf_int(alpha=0.2)  # 80% interval
    fitB, fcB = arimax(yB, H, xB, np.r_[r.saibor_3m.iloc[-1], np.repeat(s0, H - 1)], return_fit=True)  # x = SAIBOR lagged 1q
    ciB = fcB.conf_int(alpha=0.2)

    fc = pd.DataFrame(index=FC_INDEX)
    fc["mtg_real_sa_naive"] = np.exp(naive(yA, H))
    fc["mtg_real_sa_ets"] = np.exp(ets(yA, H))
    fc["mtg_real_sa_arimax"] = np.exp(fcA.predicted_mean)
    fc["mtg_real_sa_arimax_lo80"] = np.exp(ciA[:, 0])
    fc["mtg_real_sa_arimax_hi80"] = np.exp(ciA[:, 1])
    fc.round(1).to_csv(TABLES / "forecast_base.csv")

    fr = pd.DataFrame(index=pd.date_range(q_end(win.rpi_end, 1), periods=H, freq="QE"))  # RPI has its own calendar
    fr["rpi_res_naive"] = np.exp(naive(yB, H))
    fr["rpi_res_ets"] = np.exp(ets(yB, H))
    fr["rpi_res_arimax"] = np.exp(fcB.predicted_mean)
    fr["rpi_res_arimax_lo80"] = np.exp(ciB[:, 0])
    fr["rpi_res_arimax_hi80"] = np.exp(ciB[:, 1])
    fr.round(1).to_csv(TABLES / "forecast_rpi.csv")

    coefA = dict(beta_saibor=fitA.params[0], se=fitA.bse[0], p=fitA.pvalues[0], ar1=fitA.params[1])
    coefB = dict(beta_saibor_lag1=fitB.params[0], se=fitB.bse[0], p=fitB.pvalues[0], ar1=fitB.params[1])
    pd.DataFrame([dict(model="A mortgage", **coefA), dict(model="B RPI", **{k.replace('_lag1', ''): v for k, v in coefB.items()})]).round(4)\
        .to_csv(TABLES / "arimax_coefficients.csv", index=False)

    pd.set_option("display.width", 200)
    print("Backtest (RMSE/MAE in ~%):\n", scores.pivot_table(index=["target", "model"], columns="h", values="RMSE").round(1).to_string())
    print("\nNSA baselines RMSE (~%):\n", nsa.to_string())
    print("\nSAIBOR base level:", round(s0, 3))
    print("ARIMAX A:", {k: round(float(v), 4) for k, v in coefA.items()})
    print("ARIMAX B:", {k: round(float(v), 4) for k, v in coefB.items()})
    print("\nForecast (mortgages):\n", fc.round(1).T.to_string())
    print("\nForecast (RPI residential):\n", fr.round(1).T.to_string())
    print("\nLast actual: real SA mortgage", round(m.mtg_real_sa.iloc[-1], 1), "| RPI res", r.rpi_residential.iloc[-1])
    return dict(fitA=fitA, fitB=fitB, s0=s0)


if __name__ == "__main__":
    main()
