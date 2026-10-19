"""Run the full pipeline 01 -> 06 (python run_all.py)."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for step in ["01_acquisition.py", "02_harmonize.py", "03_eda.py", "04_relationships.py", "05_forecast.py", "06_scenarios.py"]:
    print(f"\n{'=' * 20} {step} {'=' * 20}")
    runpy.run_path(str(ROOT / step), run_name="__main__")
