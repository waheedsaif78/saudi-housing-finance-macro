"""Data-integrity tests: run after 01/02 (pytest -q tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shfm.common import PROCESSED, TABLES  # noqa: E402

pytestmark = pytest.mark.skipif(not (PROCESSED / "quarterly.parquet").exists(), reason="run 01/02 first")


@pytest.fixture(scope="module")
def q() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "quarterly.parquet")


def test_monthly_sums_match_sama_quarterly() -> None:
    chk = pd.read_csv(TABLES / "check_monthly_vs_sama_quarterly.csv")
    assert chk.value_gap_pct.abs().max() < 0.01
    assert chk.contracts_gap_pct.abs().max() < 0.01


def test_kapsarc_copy_matches_gastat() -> None:
    k = pd.read_csv(TABLES / "check_kapsarc_vs_gastat_rpi.csv")
    assert len(k) > 50 and k.abs_diff.max() < 0.01


def test_published_q2_2026_headlines(q: pd.DataFrame) -> None:
    """Regression check against GASTAT's published Q2-2026 release (fails if GASTAT later revises history)."""
    # GASTAT Q2-2026: general +1.3% YoY, residential plot +6.3%, villa -9.7%
    yoy = 100 * (q.loc["2026-06-30"] / q.loc["2025-06-30"] - 1)
    assert yoy["rpi_general"] == pytest.approx(1.3, abs=0.06)
    assert yoy["rpi_res_plot"] == pytest.approx(6.3, abs=0.06)
    assert yoy["rpi_villa"] == pytest.approx(-9.7, abs=0.06)


def test_villa_share_2024_matches_argaam(q: pd.DataFrame) -> None:
    """Argaam: villas ~64% of BANK residential mortgage value in 2024."""
    y = q.loc["2024-03-31":"2024-12-31"]
    share = 100 * y.mortgage_bank_value_villa.sum() / y.mortgage_bank_value_total.sum()
    assert share == pytest.approx(64.0, abs=0.5)


def test_mix_components_add_up(q: pd.DataFrame) -> None:
    x = q.loc["2018-03-31":"2026-06-30"]
    parts = x[["mtg_value_villa", "mtg_value_apartment", "mtg_value_land"]].sum(axis=1)
    assert np.allclose(parts, x.mtg_value_total, rtol=0.01)


def test_splice_continuity(q: pd.DataFrame) -> None:
    assert q.loc["2021-03-31", "rpi_spliced_general"] == pytest.approx(q.loc["2021-03-31", "rpi_general"])


def test_windows_follow_the_data() -> None:
    from shfm.config import windows

    long = pd.read_parquet(PROCESSED / "long.parquet")
    w = windows(long)
    assert w.analysis_end == min(w.rpi_end, w.mtg_end)
    assert w.fc_index[0] > w.mtg_end and len(w.fc_index) == 4
    assert 0 <= w.partial_months <= 2


def test_latest_release_is_selected(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    for name in ["gastat__REPI_Q4_2025.xlsx", "gastat__REPI_Q2_2026.xlsx", "gastat__REPI_Q1_2026.xlsx",
                 "gastat__CPI_Tables_Sep_2026.xlsx", "gastat__CPI_Tables_Dec_2025.xlsx"]:
        (tmp_path / name).touch()
    spec = importlib.util.spec_from_file_location("acq", ROOT / "01_acquisition.py")
    acq = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(acq)  # type: ignore[union-attr]
    monkeypatch.setattr(acq, "RAW", tmp_path)
    assert acq.latest("gastat__REPI_Q*.xlsx", r"Q(\d)_(\d{4})").name == "gastat__REPI_Q2_2026.xlsx"
    assert acq.latest("gastat__CPI_Tables_*.xlsx", r"Tables_([A-Za-z]{3})[a-z]*_(\d{4})").name == "gastat__CPI_Tables_Sep_2026.xlsx"
