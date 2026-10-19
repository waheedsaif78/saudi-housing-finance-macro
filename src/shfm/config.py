"""Analysis windows derived from the data, so the pipeline extends itself when new releases are added.

Only methodological anchors are fixed dates (start of the new RPI, the old/new overlap, policy events).
Everything that depends on "the latest data" is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RPI_NEW_START = pd.Timestamp("2021-03-31")   # GASTAT new methodology back-cast start
BREAK_OVERLAP_END = pd.Timestamp("2024-06-30")  # last quarter published under the old method
MTG_START = pd.Timestamp("2016-03-31")        # first quarter of SAMA monthly mortgage data
MIX_START = pd.Timestamp("2018-03-31")        # first quarter with villa/apartment/land split


def q_end(ts: pd.Timestamp, shift: int = 0) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("Q") + shift).to_timestamp(how="end").normalize()


@dataclass(frozen=True)
class Windows:
    rpi_end: pd.Timestamp          # last quarter of the RPI
    mtg_end: pd.Timestamp          # last COMPLETE quarter of mortgage data
    month_end: pd.Timestamp        # last month of mortgage data
    analysis_end: pd.Timestamp     # last quarter with both RPI and mortgages
    partial_months: int            # months already observed in the quarter after mtg_end (0-2)

    @property
    def fc_index(self) -> pd.DatetimeIndex:
        return pd.date_range(q_end(self.mtg_end, 1), periods=4, freq="QE")


def windows(long: pd.DataFrame) -> Windows:
    rpi = long[long.series == "rpi_general"].dropna(subset=["value"]).date.max()
    m = long[long.series == "mortgage_bank_value_total"].dropna(subset=["value"]).date
    month_end = m.max()
    counts = m.dt.to_period("Q").value_counts()
    mtg_end = max(p for p, c in counts.items() if c == 3).to_timestamp(how="end").normalize()
    partial = int(counts.get(month_end.to_period("Q"), 0)) if q_end(month_end) > mtg_end else 0
    return Windows(rpi_end=pd.Timestamp(rpi), mtg_end=mtg_end, month_end=pd.Timestamp(month_end),
                   analysis_end=min(pd.Timestamp(rpi), mtg_end), partial_months=partial)
