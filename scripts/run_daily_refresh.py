"""Daily public-data refresh. Financial and macro rebuilds remain scheduled separately."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    # Only daily sources run here. Quarterly SEC/ALFRED rebuilds should run after a new filing.
    for script in ("collect_s1.py", "collect_competitors.py", "collect_industry.py", "collect_s2.py", "collect_forum_nlp.py", "build_dataset.py", "build_dashboard.py", "sync_mysql.py", "precompute_forecasts.py", "build_forecast_nowcasts.py"):
        run(script)


if __name__ == "__main__":
    main()
