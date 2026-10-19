"""04 - Relationships: stationarity (ADF/KPSS), lagged cross-correlation (CCF), Granger tests,
mix-vs-price by property type, rate -> mortgage sensitivity, and policy step-dummies.

Primary sample: 2021Q1 -> latest common quarter (new RPI). Mortgage flows are seasonally adjusted via monthly STL
before aggregation, so quarter-on-quarter changes are not dominated by Q1/Q2 seasonality.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss  # noqa: E402

from shfm.common import PROCESSED, TABLES, save_fig  # noqa: E402
from shfm.config import MIX_START, MTG_START, RPI_NEW_START, windows  # noqa: E402

warnings.filterwarnings("ignore")


def monthly_panel() -> pd.DataFrame:
    long = pd.read_parquet(PROCESSED / "long.parquet")
    mo = long[long.freq == "M"].pivot_table(index="date", columns="series", values="value").loc["2016-01-31":windows(long).month_end]
    for part in ["value_total", "contracts", "value_villa", "value_apartment", "value_land"]:
        mo[f"mtg_{part}"] = mo[f"mortgage_bank_{part}"] + mo[f"mortgage_fin_{part}"]
    mo["mtg_real"] = mo["mtg_value_total"] / (mo["cpi_general"] / 100)
    for col, name in [("mtg_real", "mtg_real_sa"), ("mtg_contracts", "mtg_contracts_sa")]:
        y = np.log(mo[col])
        y.index.freq = "ME"
        stl = STL(y, period=12, robust=True).fit()
        mo[name] = np.exp(y - stl.seasonal)
    return mo


def to_quarterly(mo: pd.DataFrame) -> pd.DataFrame:
    g = mo.groupby(mo.index.to_period("Q"))
    full = g.size() == 3
    qs = pd.DataFrame({"mtg_real_sa": g["mtg_real_sa"].sum(), "mtg_contracts_sa": g["mtg_contracts_sa"].sum()})[full]
    qs.index = qs.index.to_timestamp(how="end").normalize()
    return qs


def stationarity(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in series.items():
        s = s.dropna()
        adf_p = adfuller(s, autolag="AIC", maxlag=min(4, len(s) // 4))[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        verdict = "stationary" if adf_p < 0.10 and kpss_p > 0.05 else ("non-stationary" if adf_p >= 0.10 and kpss_p <= 0.05 else "inconclusive")
        rows.append(dict(series=name, n=len(s), adf_p=round(adf_p, 3), kpss_p=round(kpss_p, 3), verdict=verdict))
    return pd.DataFrame(rows)


def ccf(x: pd.Series, y: pd.Series, lags: range) -> pd.DataFrame:
    """corr(x_{t-k}, y_t): positive k => x LEADS y by k periods."""
    rows = []
    for k in lags:
        pair = pd.concat([x.shift(k), y], axis=1).dropna()
        rows.append(dict(lag=k, corr=pair.corr().iloc[0, 1], n=len(pair), band95=1.96 / np.sqrt(len(pair))))
    return pd.DataFrame(rows)


def granger(df: pd.DataFrame, cause: str, effect: str, maxlag: int) -> list[dict[str, object]]:
    data = df[[effect, cause]].dropna()
    res = grangercausalitytests(data, maxlag=maxlag)
    return [dict(cause=cause, effect=effect, lag=L, F=round(r[0]["ssr_ftest"][0], 2), p=round(r[0]["ssr_ftest"][1], 3), n=len(data))
            for L, r in res.items()]


def main() -> None:
    q = pd.read_parquet(PROCESSED / "quarterly.parquet")
    win = windows(pd.read_parquet(PROCESSED / "long.parquet"))
    end = win.analysis_end
    tag = f"{RPI_NEW_START.to_period('Q')}-{end.to_period('Q')}"
    mo = monthly_panel()
    q = q.join(to_quarterly(mo))
    q.to_parquet(PROCESSED / "quarterly_sa.parquet")
    w = q.loc[RPI_NEW_START:end].copy()

    d = pd.DataFrame(index=w.index)
    d["dp_general"] = 100 * np.log(w.rpi_general).diff()
    d["dp_residential"] = 100 * np.log(w.rpi_residential).diff()
    d["dm_value"] = 100 * np.log(w.mtg_real_sa).diff()
    d["dm_contracts"] = 100 * np.log(w.mtg_contracts_sa).diff()
    d["d_saibor"] = w.saibor_3m.diff()

    # ---- 1. Stationarity
    st = pd.concat([
        stationarity({"log RPI general (Q)": np.log(w.rpi_general), "log RPI residential (Q)": np.log(w.rpi_residential),
                      "log real mortgage SA (Q)": np.log(w.mtg_real_sa), "SAIBOR 3M (Q)": w.saibor_3m}),
        stationarity({k: d[k] for k in ["dp_general", "dp_residential", "dm_value", "dm_contracts", "d_saibor"]}),
        stationarity({"log real mortgage SA (M, 2016-26)": np.log(mo.mtg_real_sa), "SAIBOR 3M (M, 2016-26)": mo.saibor_3m,
                      "dlog real mortgage SA (M)": 100 * np.log(mo.mtg_real_sa).diff(), "d SAIBOR (M)": mo.saibor_3m.diff()}),
    ])
    st.to_csv(TABLES / "stationarity.csv", index=False)

    # ---- 2. CCF: mortgage growth (leads?) vs price growth
    lags = range(-4, 5)
    c1 = ccf(d.dm_value, d.dp_residential, lags).assign(pair="real mortgage value -> RPI residential")
    c2 = ccf(d.dm_contracts, d.dp_residential, lags).assign(pair="mortgage contracts -> RPI residential")
    c3 = ccf(d.dm_value, d.dp_general, lags).assign(pair="real mortgage value -> RPI general")
    cc = pd.concat([c1, c2, c3]).round(3)
    cc.to_csv(TABLES / "ccf_mortgage_vs_rpi.csv", index=False)

    # Robustness on the long (spliced) RPI 2016 -> end - methodology caveat applies
    lq = q.loc[MTG_START:end]
    dl = pd.DataFrame({"dm": 100 * np.log(lq.mtg_real_sa).diff(), "dp": 100 * np.log(lq.rpi_spliced_residential).diff()})
    c_long = ccf(dl.dm, dl.dp, lags).round(3)
    c_long.to_csv(TABLES / "ccf_long_spliced_robustness.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, df, title in [(axes[0], c1, f"New RPI, {tag}"), (axes[1], c_long, f"Spliced RPI, 2016-{end.to_period('Q')} (robustness)")]:
        ax.bar(df.lag, df["corr"], color=["#2563eb" if k > 0 else "#9ca3af" for k in df.lag])
        ax.axhline(df.band95.iloc[0], color="r", ls="--", lw=0.8)
        ax.axhline(-df.band95.iloc[0], color="r", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("k (quarters): mortgage growth at t-k vs residential price growth at t")
    axes[0].set_ylabel("correlation")
    save_fig(fig, "fig07_ccf_mortgage_rpi")

    # ---- 3. Granger (on differences)
    gr = []
    for cause, effect in [("dm_value", "dp_residential"), ("dp_residential", "dm_value"),
                          ("dm_contracts", "dp_residential"), ("dp_residential", "dm_contracts"),
                          ("dm_value", "dp_general")]:
        gr += granger(d, cause, effect, maxlag=2)
    mg = pd.DataFrame({"dm": 100 * np.log(mo.mtg_real_sa).diff(), "ds": mo.saibor_3m.diff()})
    gr += [dict(r, sample=f"monthly 2016-{win.month_end.year}") for r in granger(mg, "ds", "dm", maxlag=6)]
    mg21 = mg.loc["2021-01-31":]
    gr += [dict(r, sample=f"monthly 2021-{win.month_end.year}") for r in granger(mg21, "ds", "dm", maxlag=6)]
    grd = pd.DataFrame(gr)
    grd["sample"] = grd["sample"].fillna(f"quarterly {tag}")
    grd.to_csv(TABLES / "granger.csv", index=False)

    # ---- 4. Rates -> mortgage volume: distributed-lag on YoY (quarterly, 2017Q1 -> end), HAC SE
    rq = q.loc[MTG_START:win.mtg_end].copy()
    rq["g_val"] = 100 * np.log(rq.mtg_real_sa).diff(4)
    rq["g_cnt"] = 100 * np.log(rq.mtg_contracts_sa).diff(4)
    rq["ds4"] = rq.saibor_3m.diff(4)
    rate_rows = []
    for dep in ["g_val", "g_cnt"]:
        for start in ["2017-03-31", "2021-03-31"]:
            for k in range(0, 5):
                sub = rq.loc[start:, [dep, "ds4"]].assign(x=lambda f, k=k: f.ds4.shift(k)).dropna()
                fit = sm.OLS(sub[dep], sm.add_constant(sub["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
                rate_rows.append(dict(dep=dep, sample_from=start, lag_q=k, beta_pct_per_100bp=fit.params["x"],
                                      p=fit.pvalues["x"], r2=fit.rsquared, n=int(fit.nobs)))
    rr = pd.DataFrame(rate_rows).round(3)
    rr.to_csv(TABLES / "rate_sensitivity_yoy.csv", index=False)

    # ---- 5. Mix vs prices by type (Q2): YoY of financing value by type vs YoY of matching RPI type
    pairs = {"villa": "villa", "apartment": "apartment", "land": "res_plot"}
    panel, rows = [], []
    for t, p in pairs.items():
        sub = pd.DataFrame({"g_fin": 100 * np.log(q[f"mtg_real_value_{t}"]).diff(4),
                            "g_price": 100 * np.log(q[f"rpi_{p}"]).diff(4),
                            "d_share": q[f"share_{t}"].diff(4)}).loc[RPI_NEW_START + pd.offsets.QuarterEnd(4):end].dropna()
        for k in [0, 1, 2]:
            pair = pd.concat([sub.g_fin.shift(k), sub.g_price], axis=1).dropna()
            pair2 = pd.concat([sub.d_share.shift(k), sub.g_price], axis=1).dropna()
            rows.append(dict(type=t, lag_q=k, corr_fin_growth_vs_price_growth=pair.corr().iloc[0, 1],
                             corr_share_change_vs_price_growth=pair2.corr().iloc[0, 1], n=len(pair)))
        panel.append(sub.assign(type=t))
    mix = pd.DataFrame(rows).round(3)
    mix.to_csv(TABLES / "mix_vs_price_by_type.csv", index=False)
    pan = pd.concat(panel)
    fe = sm.OLS(pan.g_price, sm.add_constant(pd.concat([pan.g_fin, pd.get_dummies(pan.type, drop_first=True, dtype=float)], axis=1))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(pan.index)[0]})

    # ---- 6. Policy dummies (step dummies; see 02_harmonize.EVENTS) - indicative only, n=21
    dd = d.join(w[["d_land_reform_2025q2", "d_rent_freeze_2025q4"]]).dropna()
    dd["dm_value_l1"] = dd.dm_value.shift(1)
    dum_rows = []
    for dep, xs in [("dp_residential", ["dm_value_l1", "d_saibor", "d_land_reform_2025q2", "d_rent_freeze_2025q4"]),
                    ("dm_value", ["d_saibor", "d_land_reform_2025q2", "d_rent_freeze_2025q4"])]:
        sub = dd[[dep, *xs]].dropna()
        fit = sm.OLS(sub[dep], sm.add_constant(sub[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        dum_rows += [dict(dep=dep, var=v, coef=fit.params[v], p=fit.pvalues[v], n=int(fit.nobs), r2=fit.rsquared) for v in xs]
    dum = pd.DataFrame(dum_rows).round(3)
    dum.to_csv(TABLES / "policy_dummies_ols.csv", index=False)

    # share levels table (annual)
    yr = q.loc[MIX_START:win.mtg_end, ["mtg_value_villa", "mtg_value_apartment", "mtg_value_land"]]
    ann = yr.groupby(yr.index.year).sum()
    ann = (100 * ann.div(ann.sum(axis=1), axis=0)).round(1)
    ann.columns = ["villa_%", "apartment_%", "land_%"]
    nq = yr.groupby(yr.index.year).size()
    ann.index = [f"{y}{f' ({nq[y]}Q)' if nq[y] < 4 else ''}" for y in ann.index]
    ann.to_csv(TABLES / "mix_shares_annual.csv")

    pd.set_option("display.width", 200)
    print("STATIONARITY\n", st.to_string(index=False))
    print("\nCCF (primary)\n", cc[cc.pair.str.contains("value -> RPI residential|contracts")].to_string(index=False))
    print("\nCCF long spliced\n", c_long.to_string(index=False))
    print("\nGRANGER\n", grd.to_string(index=False))
    print("\nRATE SENSITIVITY\n", rr.to_string(index=False))
    print("\nMIX vs PRICE\n", mix.to_string(index=False))
    print(f"\nPooled FE: price YoY on financing YoY beta={fe.params['g_fin']:.3f} p={fe.pvalues['g_fin']:.3f} n={int(fe.nobs)}")
    print("\nPOLICY DUMMIES (QoQ %, HAC)\n", dum.to_string(index=False))
    print("\nAnnual mix shares\n", ann.to_string())


if __name__ == "__main__":
    main()
