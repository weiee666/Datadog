"""Precompute the dashboard's default forecast configuration after each refresh."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.ddog_api import (  # noqa: E402
    FORECAST_DEFAULT_FACTORS,
    FORECAST_TARGETS,
    database,
    forecast_cache_key,
    forecast_payload,
)


def main() -> None:
    factors = list(FORECAST_DEFAULT_FACTORS)
    results = {}
    for target_name in FORECAST_TARGETS:
        results[target_name] = forecast_payload(factors, target_name, 0, use_cache=False)

    with database() as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS forecast_cache ("
            "cache_key VARCHAR(512) PRIMARY KEY, target_name VARCHAR(64) NOT NULL, "
            "factors_json TEXT NOT NULL, target_lags TINYINT NOT NULL, "
            "payload_json LONGTEXT NOT NULL, computed_at DATETIME NOT NULL) "
            "CHARACTER SET utf8mb4"
        )
        for target_name, payload in results.items():
            cur.execute(
                "REPLACE INTO forecast_cache "
                "(cache_key, target_name, factors_json, target_lags, payload_json, computed_at) "
                "VALUES (%s, %s, %s, %s, %s, UTC_TIMESTAMP())",
                (
                    forecast_cache_key(factors, target_name, 0),
                    target_name,
                    json.dumps(factors, ensure_ascii=False),
                    0,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        conn.commit()

    print(f"precomputed {len(results)} default forecasts at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
