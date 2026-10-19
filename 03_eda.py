"""03 - EDA: series plots, YoY table, STL of monthly mortgage flow, Ramadan & summer effects."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

from shfm.common import PROCESSED, TABLES, save_fig  # noqa: E402
from shfm.config import BREAK_OVERLAP_END, MIX_START, MTG_START, RPI_NEW_START, windows  # noqa: E402

plt.rcParams.update({"figure.figsize": (10, 4.8), "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
C = {"general": "#1f2937", "res_plot": "#2563eb", "villa": "#dc2626", "apartment": "#059669", "land": "#2563eb"}

# 1 Ramadan (Umm al-Qura), approximate to +/-1 day; 30-day window assumed. 2027-2028 are projected dates.
RAMADAN_START = ["2016-06-06", "2017-05-27", "2018-05-16", "2019-05-06", "2020-04-24", "2021-04-13",
                 "2022-04-02", "2023-03-23", "2024-03-11", "2025-03-01", "2026-02-18", "2027-02-08", "2028-01-28"]


def ramadan_share(index: pd.DatetimeIndex) -> pd.Series:
    days = pd.Series(0.0, index=index)
    for s in RAMADAN_START:
        for d in pd.date_range(s, periods=30, freq="D"):
            me = d.to_period("M").to_timestamp(how="end").normalize()
            if me in days.index:
                days[me] += 1
    return days / index.days_in_month


def main() -> None:
    q = pd.read_parquet(PROCESSED / "quarterly.parquet")
    long = pd.read_parquet(PROCESSED / "long.parquet")
    win = windows(long)
    w = q.loc[RPI_NEW_START:win.rpi_end]

    # ---- Fig 1: RPI by residential type (new methodology)
    fig, ax = plt.subplots()
    for t, lab in [("general", "General"), ("res_plot", "Residential plot"), ("villa", "Villa"), ("apartment", "Apartment")]:
        ax.plot(w.index, w[f"rpi_{t}"], label=lab, color=C[t], lw=2.2 if t == "general" else 1.6)
    ax.set_title("GASTAT Real Estate Price Index (new method, 2023=100)")
    ax.legend(frameon=False, ncol=4)
    save_fig(fig, "fig01_rpi_by_type")

    # ---- Fig 2: residential new mortgages (banks + finance cos), nominal vs real, + contracts
    m = q.loc[MTG_START:win.mtg_end]
    fig, ax = plt.subplots()
    ax.bar(m.index, m["mtg_value_total"] / 1000, width=60, color="#93c5fd", label="Nominal value (SAR bn)")
    ax.plot(m.index, m["mtg_real_value_total"] / 1000, color="#1d4ed8", lw=2, label="Real value, 2023 prices (SAR bn)")
    ax2 = ax.twinx()
    ax2.plot(m.index, m["mtg_contracts"] / 1000, color="#f59e0b", lw=1.6, ls="--", label="Contracts ('000, rhs)")
    ax2.spines["top"].set_visible(False)
    ax.set_title("New residential mortgages to individuals - quarterly (banks + finance companies)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    save_fig(fig, "fig02_mortgage_quarterly")

    # ---- Fig 3: finance mix vs price change by type
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    mm = q.loc[MIX_START:win.mtg_end]
    for t in ["villa", "apartment", "land"]:
        axes[0].plot(mm.index, mm[f"share_{t}"], label=t.capitalize(), color=C[t if t != "land" else "res_plot"], lw=1.8)
    axes[0].set_title("Mix of new mortgage value (%)")
    axes[0].legend(frameon=False)
    for t, lab in [("villa", "Villa"), ("apartment", "Apartment"), ("res_plot", "Residential plot")]:
        axes[1].plot(w.index, 100 * w[f"rpi_{t}"].pct_change(4), label=lab, color=C[t], lw=1.8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("RPI YoY change by type (%)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(frameon=False)
    save_fig(fig, "fig03_mix_vs_prices")

    # ---- Fig 4: rates vs mortgage growth
    mo = long[long.freq == "M"].pivot_table(index="date", columns="series", values="value")
    mo["mtg_value_total"] = mo["mortgage_bank_value_total"] + mo["mortgage_fin_value_total"]
    mo["mtg_contracts"] = mo["mortgage_bank_contracts"] + mo["mortgage_fin_contracts"]
    fig, ax = plt.subplots()
    r = mo.loc["2018-01-31":]
    ax.plot(r.index, r["repo_rate"], color="#111827", lw=1.8, label="Repo rate")
    ax.plot(r.index, r["saibor_3m"], color="#7c3aed", lw=1.6, label="3M SAIBOR")
    ax2 = ax.twinx()
    yoy = 100 * (r["mtg_value_total"].rolling(3).sum() / r["mtg_value_total"].rolling(3).sum().shift(12) - 1)
    ax2.plot(r.index, yoy, color="#f97316", lw=1.2, alpha=0.8, label="Mortgage value YoY, 3m sum (%, rhs)")
    ax2.axhline(0, color="#f97316", lw=0.6, ls=":")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_title("Policy & interbank rates vs new-mortgage growth (monthly)")
    save_fig(fig, "fig04_rates_vs_mortgages")

    # ---- Fig 5: methodology break
    fig, ax = plt.subplots()
    lq = q.loc[MTG_START:win.rpi_end]
    k = lq.loc[RPI_NEW_START, "rpi_general"] / lq.loc[RPI_NEW_START, "rpi_old_general"]
    pre, post = lq.loc[:RPI_NEW_START], lq.loc[RPI_NEW_START:BREAK_OVERLAP_END]
    ax.plot(pre.index, pre["rpi_old_general"] * k, color="#9ca3af", lw=1.6, label="Old method (2014=100), rescaled at 2021Q1")
    ax.plot(post.index, post["rpi_old_general"] * k, color="#9ca3af", lw=1.2, ls="--", label="Old method 2021Q1-2024Q2 (overlap)")
    ax.plot(w.index, w["rpi_general"], color="#111827", lw=2.2, label="New method (2023=100)")
    ax.axvline(RPI_NEW_START, color="k", lw=0.8, ls=":")
    ax.set_title("RPI general index: old vs new methodology (aligned at 2021Q1)")
    ax.legend(frameon=False)
    save_fig(fig, "fig05_rpi_methodology_break")

    # ---- YoY table (primary window)
    yoy_cols = {"rpi_general": "RPI general", "rpi_residential": "RPI residential", "rpi_res_plot": "RPI res. plot",
                "rpi_villa": "RPI villa", "rpi_apartment": "RPI apartment", "mtg_value_total": "Mortgage value (nominal)",
                "mtg_real_value_total": "Mortgage value (real)", "mtg_contracts": "Mortgage contracts",
                "mtg_value_villa": "Mortgage value villa", "mtg_value_apartment": "Mortgage value apartment",
                "mtg_value_land": "Mortgage value land", "cpi_rent_actual": "CPI actual rents"}
    yoy = 100 * q[list(yoy_cols)].pct_change(4, fill_method=None)
    yt = yoy.loc[RPI_NEW_START + pd.offsets.QuarterEnd(4):win.analysis_end].rename(columns=yoy_cols).round(1)
    yt["SAIBOR 3M (level)"] = q["saibor_3m"].round(2)
    yt.to_csv(TABLES / "yoy_table.csv")

    # ---- STL on monthly log real mortgage value + Ramadan / summer effects
    cpi = mo["cpi_general"] / 100
    y = np.log((mo["mtg_value_total"] / cpi).loc["2016-01-31":win.month_end]).dropna()
    y.index.freq = "ME"
    stl = STL(y, period=12, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"STL: log real monthly new-mortgage value (2016-01 to {win.month_end:%Y-%m})", y=1.01)
    save_fig(fig, "fig06_stl_monthly_mortgages")

    detr = (y - stl.trend).rename("detrended")
    X = pd.get_dummies(detr.index.month, prefix="m", drop_first=True, dtype=float)
    X.index = detr.index
    X["ramadan_share"] = ramadan_share(detr.index)
    res = sm.OLS(detr, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    seas = pd.DataFrame({"month": range(1, 13),
                         "avg_seasonal_pct": [100 * (np.exp(stl.seasonal[stl.seasonal.index.month == k].mean()) - 1) for k in range(1, 13)]})
    seas.to_csv(TABLES / "seasonal_profile_monthly.csv", index=False)
    ram = dict(coef=res.params["ramadan_share"], se=res.bse["ramadan_share"], p=res.pvalues["ramadan_share"],
               effect_full_month_pct=100 * (np.exp(res.params["ramadan_share"]) - 1), n=int(res.nobs))
    pd.DataFrame([ram]).round(4).to_csv(TABLES / "ramadan_effect.csv", index=False)
    print("Seasonal profile (% vs trend):\n", seas.round(1).to_string(index=False))
    print("\nRamadan effect (full Ramadan month vs same calendar month without Ramadan):", {k: round(v, 3) for k, v in ram.items()})
    print("\nYoY table (last 6 q):\n", yt.tail(6).T.to_string())
    strength = 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal)
    print(f"\nSTL seasonal strength: {strength:.2f}")


if __name__ == "__main__":
    main()
