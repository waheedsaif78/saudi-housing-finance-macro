"""06 - Scenarios: SAIBOR flat / -50bp / +50bp over the next 4 quarters, effect on new-mortgage volume.

Method
  * Baseline: nowcast of the first forecast quarter from months already observed (scaled by the same
    quarter last year), then Naive
    (flat real SA level) - Naive was the best model in the rolling-origin backtest.
  * Rate effect = exp(beta * delta_SAIBOR), applied from the first quarter whose rates are not yet observed.
      central beta : ARIMAX(1,1,0)+SAIBOR level coefficient (05_forecast)
      range        : distributed-lag YoY estimates (04_relationships), 2021+ and 2017+ samples, lag 0
  * Real SA -> nominal NSA: x average quarterly seasonal factor (2018 -> latest) x CPI path
    (CPI grows at its latest 12-month rate).
  * Prices: no statistically detectable mortgage->price or rate->price link (04), so RPI is shown
    as the model range, unchanged across rate scenarios - stated, not assumed away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shfm.common import PROCESSED, TABLES, save_fig  # noqa: E402
from shfm.config import MIX_START, q_end, windows  # noqa: E402

SCEN = {"Flat (base)": 0.0, "-50bp": -0.5, "+50bp": +0.5}


def main() -> pd.DataFrame:
    q = pd.read_parquet(PROCESSED / "quarterly_sa.parquet")
    long = pd.read_parquet(PROCESSED / "long.parquet")
    win = windows(long)
    FC = win.fc_index
    coef = pd.read_csv(TABLES / "arimax_coefficients.csv").set_index("model")
    rs = pd.read_csv(TABLES / "rate_sensitivity_yoy.csv")
    base_fc = pd.read_csv(TABLES / "forecast_base.csv", index_col=0, parse_dates=True)

    beta_c = float(coef.loc["A mortgage", "beta_saibor"])  # log-points per 1pp
    dl = rs[(rs.dep == "g_val") & (rs.lag_q == 0)].set_index("sample_from").beta_pct_per_100bp / 100
    beta_lo, beta_hi = float(dl.max()), float(dl.min())  # least / most negative

    hist = q.loc[MIX_START:win.mtg_end]
    sf = (hist.mtg_real_value_total / hist.mtg_real_sa).groupby(hist.index.quarter).mean()
    cpi_m = long[long.series == "cpi_general"].set_index("date").value.dropna()
    g12 = cpi_m.iloc[-1] / cpi_m.iloc[-13] - 1
    cpi_q = q.cpi_general.dropna()
    cpi0 = cpi_q.loc[FC[0]] if FC[0] in cpi_q.index else cpi_q.iloc[-1] * (1 + g12) ** (((FC[0] - cpi_q.index[-1]).days / 91) / 4)
    cpi_path = pd.Series([cpi0 * (1 + g12) ** (k / 4) for k in range(4)], index=FC)

    # Nowcast the first forecast quarter from months already observed (if any), scaled by the same
    # quarter last year; then hold that seasonally adjusted real level flat (Naive) for the rest.
    mo = long[long.series.isin(["mortgage_bank_value_total", "mortgage_fin_value_total"])].groupby("date").value.sum()
    k = win.partial_months
    last4 = q.loc[q_end(win.mtg_end, -3):win.mtg_end, "mtg_value_total"].sum()
    if k > 0:
        obs = mo.loc[FC[0] - pd.offsets.QuarterEnd(1) + pd.offsets.MonthEnd(1):win.month_end]
        obs_ly = mo.loc[obs.index - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)]
        nowcast = obs.sum() * q.loc[q_end(FC[0], -4), "mtg_value_total"] / obs_ly.sum()
        level_sa = nowcast / (sf.loc[FC[0].quarter] * cpi_path.iloc[0] / 100)
        obs_txt = f"{k} month(s) observed: {obs.sum():,.0f} vs {obs_ly.sum():,.0f} a year earlier ({100*(obs.sum()/obs_ly.sum()-1):+.1f}%)"
        shift_mask = np.array([0, 1, 1, 1])  # rates for the nowcast quarter are already (partly) observed
    else:
        level_sa = base_fc["mtg_real_sa_naive"].iloc[0]
        nowcast, obs_txt = float("nan"), "no months observed in the first forecast quarter"
        shift_mask = np.ones(4)
    base_real = np.repeat(level_sa, 4)
    rows = []
    for name, ds in SCEN.items():
        for lab, b in [("central", beta_c), ("low_sens", beta_lo), ("high_sens", beta_hi)]:
            real_sa = base_real * np.exp(b * ds * shift_mask)
            nominal = real_sa * sf.loc[FC.quarter].values * cpi_path.values / 100
            for d, rsa, nom in zip(FC, real_sa, nominal, strict=True):
                rows.append(dict(scenario=name, beta=lab, date=d, real_sa=rsa, nominal_nsa=nom))
    sc = pd.DataFrame(rows)
    sc.round({"real_sa": 1, "nominal_nsa": 1}).to_csv(TABLES / "scenarios_quarterly.csv", index=False)

    summ = sc.groupby(["scenario", "beta"]).nominal_nsa.sum().unstack()
    summ = summ[["high_sens", "central", "low_sens"]].rename(columns=lambda c: f"4Q total SAR mn ({c})")
    summ["central vs last 4Q actual %"] = 100 * (summ["4Q total SAR mn (central)"] / last4 - 1)
    summ["central vs flat %"] = 100 * (summ["4Q total SAR mn (central)"] / summ.loc["Flat (base)", "4Q total SAR mn (central)"] - 1)
    summ = summ.loc[list(SCEN)]
    # Reference: seasonal-naive path (nowcast + same quarters of last year), a more conservative baseline
    sn = [q.loc[q_end(d, -4), "mtg_value_total"] for d in FC]
    if k > 0:
        sn[0] = nowcast
    ref_seasonal_naive = float(np.sum(sn))
    pd.DataFrame([dict(first_fc_quarter=FC[0].date(), months_observed=k, nowcast=nowcast, last4q_actual=last4,
                       seasonal_naive_4q=ref_seasonal_naive)]).round(0).to_csv(TABLES / "nowcast_reference.csv", index=False)
    summ.round(1).to_csv(TABLES / "scenarios_summary.csv")

    # ---- Figure: history + scenarios (nominal, NSA)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    h = q.loc[q_end(win.mtg_end, -17):win.mtg_end, "mtg_value_total"] / 1000
    ax.plot(h.index, h, color="#111827", lw=2, marker="o", ms=3, label="Actual")
    colors = {"Flat (base)": "#2563eb", "-50bp": "#059669", "+50bp": "#dc2626"}
    for name in SCEN:
        s = sc[(sc.scenario == name)].pivot(index="date", columns="beta", values="nominal_nsa") / 1000
        ax.plot([h.index[-1], *s.index], [h.iloc[-1], *s["central"]], color=colors[name], lw=1.8, ls="--", label=name)
        if name != "Flat (base)":
            band = s[["low_sens", "high_sens"]]
            ax.fill_between(s.index, band.min(axis=1), band.max(axis=1), color=colors[name], alpha=0.12)
    ratio_lo = (base_fc["mtg_real_sa_arimax_lo80"] / base_fc["mtg_real_sa_arimax"]).values
    ratio_hi = (base_fc["mtg_real_sa_arimax_hi80"] / base_fc["mtg_real_sa_arimax"]).values
    lo = base_real * ratio_lo * sf.loc[FC.quarter].values * cpi_path.values / 100 / 1000
    hi = base_real * ratio_hi * sf.loc[FC.quarter].values * cpi_path.values / 100 / 1000
    ax.fill_between(FC, lo, hi, color="#9ca3af", alpha=0.15, label="80% model uncertainty (base)")
    first_shift = FC[int(np.argmax(shift_mask))].to_period("Q")
    ax.set_title(f"New residential mortgages (SAR bn, quarterly): SAIBOR scenarios from {first_shift}")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save_fig(fig, "fig08_scenarios_mortgage")

    pd.set_option("display.width", 220)
    print(f"beta central {beta_c:.4f}, DL range [{beta_lo:.3f}, {beta_hi:.3f}] per 1pp | CPI 12m growth {100*g12:.2f}%")
    print("seasonal factors by quarter:", sf.round(3).to_dict())
    print(f"\nScenario summary (nominal, {FC[0].to_period('Q')}-{FC[-1].to_period('Q')}):\n", summ.round(1).to_string())
    print(f"Seasonal-naive reference 4Q total: {ref_seasonal_naive:,.0f} ({100*(ref_seasonal_naive/last4-1):+.1f}% vs last 4Q)")
    print(f"Last 4Q actual (to {win.mtg_end.to_period('Q')}): {last4:,.0f} SAR mn | nowcast {FC[0].to_period('Q')}: {nowcast:,.0f} ({obs_txt})")
    central = sc[sc.beta == "central"].pivot(index="date", columns="scenario", values="nominal_nsa")
    print("\nQuarterly central:\n", central.round(0).to_string())
    return sc


if __name__ == "__main__":
    main()
