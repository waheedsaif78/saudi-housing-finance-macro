"""02 - Harmonize: monthly -> quarterly, real (CPI-deflated) finance, mix shares, RPI splice, policy dummies.

Rules
  * Flows (mortgage value, contracts): quarterly SUM; only complete quarters (3 months) are kept.
  * Rates and CPI: quarterly MEAN.
  * Real values: nominal / (CPI general / 100)  -> SAR mn at 2023 prices.
  * Total market = banks + finance companies.
  * RPI methodological break (GASTAT new method from Q3-2024, base 2023, back-cast to 2021Q1):
      - PRIMARY analysis uses the new series only (2021Q1 -> latest quarter).
      - A ratio-spliced long series (old 2014=100 rescaled at the 2021Q1 overlap) is built
        for robustness only, and the two methods' divergence over 2021-2024 is documented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shfm.common import PROCESSED, TABLES  # noqa: E402
from shfm.config import BREAK_OVERLAP_END, RPI_NEW_START, windows  # noqa: E402

SPLICE_Q = RPI_NEW_START

# Policy events (dates verified from public sources; see README / 07_insights.md)
EVENTS = {
    # RETT 5% replaces VAT on property, first-home RETT covered by state up to SAR1m - effective 4-Oct-2020
    "d_rett_2020q4": "2020-10-04",
    # Crown Prince directives on Riyadh land supply (29-Mar-2025) + White-Land law amendments
    # approved by Cabinet 29-Apr-2025 -> combined into one step dummy (they are not separable quarterly)
    "d_land_reform_2025q2": "2025-04-01",
    # Riyadh residential & commercial rent freeze for 5 years, effective 25-Sep-2025
    "d_rent_freeze_2025q4": "2025-10-01",
}


def quarterly(long: pd.DataFrame) -> pd.DataFrame:
    m = long[long.freq == "M"].copy()
    m["q"] = m.date.dt.to_period("Q")
    flows = m.series.str.startswith("mortgage_")
    agg_sum = m[flows].groupby(["q", "series"]).agg(value=("value", lambda s: s.sum(min_count=1)), n=("value", "size"))
    agg_sum = agg_sum[agg_sum.n == 3].value.unstack()
    agg_mean = m[~flows].groupby(["q", "series"]).agg(value=("value", "mean"), n=("value", "size"))
    agg_mean = agg_mean[agg_mean.n == 3].value.unstack()
    qdf = long[long.freq == "Q"].assign(q=lambda d: d.date.dt.to_period("Q")).pivot_table(index="q", columns="series", values="value")
    out = agg_sum.join(agg_mean, how="outer").join(qdf, how="outer").sort_index()
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out


def enrich(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    for part in ["contracts", "value_total", "value_villa", "value_apartment", "value_land"]:
        q[f"mtg_{part}"] = q[f"mortgage_bank_{part}"] + q[f"mortgage_fin_{part}"]
    defl = q["cpi_general"] / 100.0
    for part in ["value_total", "value_villa", "value_apartment", "value_land"]:
        q[f"mtg_real_{part}"] = q[f"mtg_{part}"] / defl
    q["mtg_avg_ticket_k"] = 1000 * q["mtg_value_total"] / q["mtg_contracts"]  # SAR thousand per contract
    split = q[["mtg_value_villa", "mtg_value_apartment", "mtg_value_land"]].sum(axis=1, min_count=3)
    for t in ["villa", "apartment", "land"]:
        q[f"share_{t}"] = 100 * q[f"mtg_value_{t}"] / split
    q["bank_share_value"] = 100 * q["mortgage_bank_value_total"] / q["mtg_value_total"]
    q["real_saibor"] = q["saibor_3m"] - 100 * np.log(q["cpi_general"] / q["cpi_general"].shift(4))

    # ---- RPI splice (ratio at 2021Q1)
    for t in ["general", "residential", "res_plot", "villa", "apartment"]:
        new, old = q[f"rpi_{t}"], q[f"rpi_old_{t}"]
        factor = new.loc[SPLICE_Q] / old.loc[SPLICE_Q]
        q[f"rpi_spliced_{t}"] = np.where(q.index < SPLICE_Q, old * factor, new)

    for name, d in EVENTS.items():
        q[name] = (q.index >= pd.Timestamp(d)).astype(int)
    return q


def break_report(q: pd.DataFrame) -> pd.DataFrame:
    """How differently old vs new methodology measured 2021Q1->2024Q2 (both available)."""
    a, b = RPI_NEW_START, BREAK_OVERLAP_END
    rows = []
    for t in ["general", "residential", "res_plot", "villa", "apartment"]:
        rows.append(dict(type=t,
                         old_method_pct=100 * (q.loc[b, f"rpi_old_{t}"] / q.loc[a, f"rpi_old_{t}"] - 1),
                         new_method_pct=100 * (q.loc[b, f"rpi_{t}"] / q.loc[a, f"rpi_{t}"] - 1),
                         corr_qoq=q.loc[a:b, [f"rpi_old_{t}", f"rpi_{t}"]].pct_change().corr().iloc[0, 1]))
    return pd.DataFrame(rows).round(2)


def main() -> pd.DataFrame:
    long = pd.read_parquet(PROCESSED / "long.parquet")
    q = enrich(quarterly(long))
    q.to_parquet(PROCESSED / "quarterly.parquet")
    br = break_report(q)
    br.to_csv(TABLES / "rpi_methodology_break.csv", index=False)
    analysis = q.loc[RPI_NEW_START:windows(long).analysis_end]
    print("quarterly panel:", q.shape, "| primary window:", analysis.index.min().date(), "->", analysis.index.max().date(), len(analysis), "quarters")
    print("\nRPI break (2021Q1->2024Q2 cumulative change):\n", br.to_string(index=False))
    cols = ["mtg_value_total", "mtg_contracts", "mtg_real_value_total", "share_villa", "share_apartment", "share_land",
            "saibor_3m", "repo_rate", "rpi_general", "rpi_residential", "cpi_rent_actual"]
    print("\n", q[cols].loc["2023-03-31":].round(1).to_string())
    return q


if __name__ == "__main__":
    main()
