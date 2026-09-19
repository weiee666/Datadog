"""Build four current-quarter forecasts after the daily source refresh.

The interactive API is intentionally read-only.  This job uses the same
quarterly factor construction as that API, selects a compact ridge model by
expanding-window MAPE, and stores one current-quarter estimate per target.
"""
from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.ddog_api import (
    FORECAST_FACTORS,
    FORECAST_TARGETS,
    database,
    number,
    quarter_of,
    solve,
    target_series_diagnostics,
)


def correlation(left, right):
    if len(left) < 6:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) *
                            sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def fit(train, ridge=0.25):
    width = len(train[0][2])
    means = [sum(row[2][column] for row in train) / len(train) for column in range(width)]
    scales = [math.sqrt(sum((row[2][column] - means[column]) ** 2 for row in train) / len(train)) or 1
              for column in range(width)]
    design = [[1.0] + [(row[2][column] - means[column]) / scales[column]
                       for column in range(width)] for row in train]
    response = [row[1] for row in train]
    size = width + 1
    gram = [[sum(row[left] * row[right] for row in design) +
             (ridge if left == right and left else 0.0) for right in range(size)]
            for left in range(size)]
    beta = solve(gram, [sum(row[column] * value for row, value in zip(design, response))
                        for column in range(size)])
    return (means, scales, beta) if beta else None


def predict(model, values):
    means, scales, beta = model
    point = [1.0] + [(value - mean) / scale for value, mean, scale in zip(values, means, scales)]
    return sum(weight * value for weight, value in zip(beta, point))


def load_features():
    with database() as conn, conn.cursor() as cur:
        cur.execute("SELECT `date`, series_id, value FROM product_daily WHERE series_id LIKE 's1.npm.%'")
        product = cur.fetchall()
        cur.execute("SELECT quarter, company, revenue, capex FROM industry_quarterly")
        industry = cur.fetchall()
        cur.execute("SELECT `date`, series_id, value FROM macro_latest WHERE series_id IN ("
                    "'macro.fred.DGS10', 'macro.fred.NFCI', 'macro.fred.PAYEMS', "
                    "'macro.fred.GDPC1', 'macro.fred.CPIAUCSL', 'macro.fred.NEWORDER', "
                    "'macro.fred.IPG3344S', 'macro.fred.GACDFSA066MSFRBPHI', "
                    "'macro.fred.GACDISA066MSFRBNY') ORDER BY `date`")
        macro = cur.fetchall()
        cur.execute("SELECT `date`, company, value FROM competitor_npm_daily ORDER BY `date`, company")
        peers = cur.fetchall()

    features = defaultdict(dict)
    npm = defaultdict(float)
    for row in product:
        npm[quarter_of(row["date"])] += number(row["value"]) or 0
    for quarter, value in npm.items():
        features[quarter]["npm_total"] = value

    cloud = defaultdict(lambda: {"revenue": 0.0, "capex": 0.0, "companies": set()})
    for row in industry:
        bucket = cloud[row["quarter"]]
        bucket["revenue"] += number(row["revenue"]) or 0
        bucket["capex"] += number(row["capex"]) or 0
        bucket["companies"].add(row["company"])
    for quarter, value in cloud.items():
        if len(value["companies"]) == 3:
            features[quarter]["cloud_revenue"] = value["revenue"]
            features[quarter]["cloud_capex"] = value["capex"]

    latest_macro = {}
    for row in macro:
        latest_macro[(quarter_of(row["date"]), row["series_id"])] = number(row["value"])
    for (quarter, series), value in latest_macro.items():
        if value is not None:
            features[quarter][f"macro_{series.rsplit('.', 1)[-1]}"] = value

    peer_totals = defaultdict(float)
    for row in peers:
        peer_totals[(quarter_of(row["date"]), str(row["company"]).lower())] += number(row["value"]) or 0
    peer_keys = {"datadog": "peer_ddog_sdk", "dynatrace": "peer_dynatrace_sdk", "elastic": "peer_elastic_sdk"}
    for (quarter, company), value in peer_totals.items():
        if company in peer_keys:
            features[quarter][peer_keys[company]] = value

    latest_signal = max([str(row["date"]) for row in product + macro + peers], default=None)
    return features, latest_signal


def target_rows(column):
    with database() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT quarter, `{column}` AS target_value FROM targets_quarterly "
                    f"WHERE `{column}` IS NOT NULL ORDER BY quarter")
        return [(row["quarter"], number(row["target_value"])) for row in cur.fetchall()]


def observations(rows, features, factors, lags, scale):
    output = []
    for index, (quarter, value) in enumerate(rows):
        values = features.get(quarter, {})
        if any(values.get(factor) is None for factor in factors) or index < lags:
            continue
        history = [rows[index - lag][1] / scale for lag in range(1, lags + 1)]
        output.append((quarter, value / scale, [values[factor] for factor in factors] + history))
    return output


def evaluate(rows, features, factors, lags, current_quarter, scale):
    samples = observations(rows, features, factors, lags, scale)
    width = len(factors) + lags
    minimum_train = max(8, width + 5)
    if len(samples) <= minimum_train:
        return None
    predictions = []
    for index in range(minimum_train, len(samples)):
        model = fit(samples[:index])
        if not model:
            continue
        quarter, actual, values = samples[index]
        predictions.append((actual, max(0.0, predict(model, values))))
    if len(predictions) < 4:
        return None
    mape = sum(abs(actual - estimate) / max(abs(actual), 1e-9) for actual, estimate in predictions) / len(predictions) * 100
    rmse = math.sqrt(sum((actual - estimate) ** 2 for actual, estimate in predictions) / len(predictions)) * scale
    current = features.get(current_quarter, {})
    if any(current.get(factor) is None for factor in factors) or len(rows) < lags:
        return None
    model = fit(samples)
    if not model:
        return None
    lag_values = [rows[-lag][1] / scale for lag in range(1, lags + 1)]
    value = max(0.0, predict(model, [current[factor] for factor in factors] + lag_values)) * scale
    return {"forecast_value": value, "mape_pct": mape, "rmse": rmse, "oos_quarters": len(predictions)}


def build_nowcast(target_name, features, latest_signal):
    target = FORECAST_TARGETS[target_name]
    rows = [(quarter, value) for quarter, value in target_rows(target["column"]) if value is not None]
    current_quarter = max(features)
    scale = 1e6 if target["unit"] == "usd" else 1.0
    current = features[current_quarter]
    available = [factor for factor in FORECAST_FACTORS if current.get(factor) is not None]
    rankings = []
    by_quarter = dict(rows)
    for factor in available:
        pairs = [(features[quarter][factor], value) for quarter, value in rows
                 if features.get(quarter, {}).get(factor) is not None]
        rankings.append((abs(correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs])), factor))
    candidates = [factor for _, factor in sorted(rankings, reverse=True)[:7]]
    diagnostics = target_series_diagnostics([
        {"quarter": quarter, "target_value": value} for quarter, value in rows
    ])
    strongest_lag = diagnostics.get("strongest_lag") or 0
    lags_to_try = sorted({0, min(2, int(strongest_lag))})
    best = None
    for lags in lags_to_try:
        for count in range(0, min(3, len(candidates)) + 1):
            for subset in itertools.combinations(candidates, count):
                result = evaluate(rows, features, list(subset), lags, current_quarter, scale)
                if result and (best is None or result["mape_pct"] < best["mape_pct"]):
                    best = {**result, "factors": list(subset), "target_lags": lags}
    if best is None:
        raise RuntimeError(f"No valid current-quarter model for {target_name}")
    return {
        "target_name": target_name,
        "target_label": target["label"],
        "unit": target["unit"],
        "quarter": current_quarter,
        "signals_through": latest_signal,
        "financial_history_through": rows[-1][0],
        "factor_names": ", ".join(best["factors"]) or "target history only",
        "target_lags": best["target_lags"],
        **best,
    }


def main():
    features, latest_signal = load_features()
    results = [build_nowcast(target, features, latest_signal) for target in FORECAST_TARGETS]
    with database() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS forecast_nowcast ("
                    "target_name VARCHAR(64) PRIMARY KEY, target_label VARCHAR(160) NOT NULL, unit VARCHAR(16) NOT NULL, "
                    "quarter VARCHAR(16) NOT NULL, forecast_value DOUBLE NOT NULL, rmse DOUBLE NULL, mape_pct DOUBLE NULL, "
                    "oos_quarters INT NULL, factor_names TEXT NULL, target_lags INT NULL, signals_through DATE NULL, "
                    "financial_history_through VARCHAR(16) NULL, computed_at DATETIME NOT NULL) CHARACTER SET utf8mb4")
        for row in results:
            cur.execute("REPLACE INTO forecast_nowcast (target_name, target_label, unit, quarter, forecast_value, rmse, "
                        "mape_pct, oos_quarters, factor_names, target_lags, signals_through, financial_history_through, computed_at) "
                        "VALUES (%(target_name)s, %(target_label)s, %(unit)s, %(quarter)s, %(forecast_value)s, %(rmse)s, "
                        "%(mape_pct)s, %(oos_quarters)s, %(factor_names)s, %(target_lags)s, %(signals_through)s, "
                        "%(financial_history_through)s, UTC_TIMESTAMP())", row)
        conn.commit()
    print(f"built {len(results)} current-quarter forecasts at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
