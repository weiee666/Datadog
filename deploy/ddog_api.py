"""Small read-only API that serves the dashboard payload from MySQL."""
from __future__ import annotations

import json
import math
import os
import gzip
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pymysql
from statsmodels.tsa.stattools import adfuller, pacf


TABLES = {
    "targets_quarterly": "基本盘 / 财务披露",
    "product_daily": "公司信号 / 产品使用",
    "industry_activity_daily": "行业 / 开源生态日频",
    "industry_activity_cumulative": "行业 / 容器与基础设施快照",
    "hiring_daily": "公司信号 / 招聘日序列",
    "hiring_jobs_snapshot": "公司信号 / 招聘快照",
    "headcount_annual": "公司信号 / 年度员工数",
    "industry_quarterly": "行业 / 云厂商",
    "macro_latest": "宏观 / 因子序列",
    "forecast_latest": "预测 / 最新结果",
    "product_downloads_fine_monthly": "公司信号 / 产品月度细分",
    "product_downloads_fine_quarterly": "公司信号 / 产品季度细分",
    "hiring_lca_fine_by_family_quarterly": "公司信号 / 招聘职位族季度细分",
    "hiring_lca_fine_by_state_quarterly": "公司信号 / 招聘州季度细分",
    "forum_mentions_daily": "公司信号 / 开发者社区日频提及",
    "forum_mentions_detail": "公司信号 / 开发者社区公开讨论明细",
    "data_runs": "运维 / 刷新记录",
}

FORECAST_FACTORS = {
    "npm_total": "npm SDK 下载量",
    "cloud_revenue": "云厂商收入",
    "cloud_capex": "云厂商资本开支",
    "macro_DGS10": "美国 10 年期国债收益率",
    "macro_NFCI": "金融条件指数 NFCI",
    "macro_PAYEMS": "美国非农就业人数",
    "macro_GDPC1": "美国实际 GDP",
    "macro_CPIAUCSL": "美国 CPI",
    "macro_NEWORDER": "核心资本品新订单",
    "macro_IPG3344S": "计算机与电子产品工业生产",
    "macro_GACDFSA066MSFRBPHI": "费城联储地区景气",
    "macro_GACDISA066MSFRBNY": "纽约联储地区景气",
    "peer_ddog_sdk": "Datadog SDK 下载量",
    "peer_dynatrace_sdk": "Dynatrace SDK 下载量",
    "peer_elastic_sdk": "Elastic SDK 下载量",
}

FORECAST_FACTOR_GROUPS = {
    "company": {"label": "公司层面", "factors": ["npm_total", "peer_ddog_sdk"]},
    "industry": {"label": "行业与竞争", "factors": [
        "peer_dynatrace_sdk", "peer_elastic_sdk", "cloud_revenue", "cloud_capex",
    ]},
    "macro_demand": {"label": "宏观需求", "factors": [
        "macro_GDPC1", "macro_PAYEMS", "macro_NEWORDER", "macro_IPG3344S",
    ]},
    "regional_activity": {"label": "区域景气", "factors": [
        "macro_GACDFSA066MSFRBPHI", "macro_GACDISA066MSFRBNY",
    ]},
    "inflation": {"label": "通胀与汇率", "factors": ["macro_CPIAUCSL"]},
    "rates_financial": {"label": "利率与金融", "factors": ["macro_DGS10", "macro_NFCI"]},
}

# The dashboard opens with every candidate factor selected and no target lag.
# That configuration is precomputed after the daily data refresh.
FORECAST_DEFAULT_FACTORS = tuple(FORECAST_FACTORS)

FORECAST_TARGETS = {
    "revenue": {"column": "revenue", "label": "季度收入（绝对值）", "unit": "usd"},
    "billings": {"column": "billings", "label": "Billings（近似口径）", "unit": "usd"},
    "rpo": {"column": "rpo", "label": "RPO", "unit": "usd"},
    "customers_100k": {"column": "customers_100k", "label": "$100k+ ARR 大客户数", "unit": "count"},
}


def target_series_diagnostics(targets):
    """Describe the target's own persistence before looking at external factors."""
    series = [(row["quarter"], number(row["target_value"])) for row in targets]
    series = [(quarter, value) for quarter, value in series if value is not None]
    quarters = [quarter for quarter, _ in series]
    levels = [value for _, value in series]

    def adf_summary(values):
        if len(values) < 8:
            return {"available": False, "reason": "样本不足（至少需要 8 个季度）"}
        try:
            maximum_lag = min(4, max(0, len(values) // 3))
            statistic, p_value, used_lag, nobs, critical, _ = adfuller(
                values, maxlag=maximum_lag, regression="c", autolag="AIC"
            )
            return {"available": True, "statistic": statistic, "p_value": p_value,
                    "used_lag": used_lag, "nobs": nobs,
                    "critical_5pct": critical.get("5%"),
                    "stationary_at_5pct": p_value < 0.05}
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            return {"available": False, "reason": str(error)}

    level_adf = adf_summary(levels)
    differences = [current - previous for previous, current in zip(levels, levels[1:])]
    difference_adf = adf_summary(differences)
    # Revenue levels trend upward over time. Comparing levels would make every
    # lag look almost perfectly correlated, so show persistence in the quarter-
    # to-quarter increment, which is the useful diagnostic for an AR lag.
    autocorrelation_values = differences

    def autocorrelation(lag):
        left, right = autocorrelation_values[lag:], autocorrelation_values[:-lag]
        if len(left) < 5:
            return None, len(left)
        left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
        numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
        denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) *
                                sum((b - right_mean) ** 2 for b in right))
        return (numerator / denominator if denominator else None), len(left)

    # PACF removes indirect effects through intervening quarters, making it
    # better suited than raw ACF for choosing AR lag terms.
    maximum_lag = min(4, max(0, len(autocorrelation_values) // 2 - 1))
    pacf_values = pacf(autocorrelation_values, nlags=maximum_lag, method="ywm") if maximum_lag else []
    lags = [{"lag": lag, "correlation": float(pacf_values[lag]),
             "n": len(autocorrelation_values) - lag}
            for lag in range(1, maximum_lag + 1)]
    valid = [row for row in lags if row["correlation"] is not None]
    strongest = max(valid, key=lambda row: abs(row["correlation"])) if valid else None
    return {"level_adf": level_adf, "difference_adf": difference_adf,
            "autocorrelation_basis": "季度新增值（本季度减上季度）",
            "partial_autocorrelation": lags,
            "strongest_lag": strongest["lag"] if strongest else None,
            "strongest_correlation": strongest["correlation"] if strongest else None}


def database():
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ.get("MYSQL_DATABASE", "ddog_dashboard"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def live_payload():
    """Build the existing dashboard contract directly from production tables."""
    with database() as conn, conn.cursor() as cur:
        cur.execute("SELECT `date`, series_id, value FROM product_daily "
                    "WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) ORDER BY `date`")
        product = cur.fetchall()
        cur.execute("SELECT * FROM targets_quarterly ORDER BY period_end")
        targets = cur.fetchall()
        cur.execute("SELECT `date`, series_id, value FROM hiring_daily ORDER BY `date`")
        hiring_daily = cur.fetchall()
        cur.execute("SELECT `function`, geography FROM hiring_jobs_snapshot")
        jobs = cur.fetchall()
        cur.execute("SELECT quarter, company, revenue, capex FROM industry_quarterly ORDER BY quarter")
        industry = cur.fetchall()
        cur.execute("SELECT `date`, category, value FROM industry_activity_daily "
                    "WHERE source = 'npm' ORDER BY `date`")
        industry_activity = cur.fetchall()
        cur.execute("SELECT `date`, series_id, value FROM macro_latest ORDER BY `date`")
        macro = cur.fetchall()
        # `forecast_nowcast` is rebuilt after each daily source refresh.  The
        # older `forecast_latest` table only contained a revenue-only snapshot
        # from the original research notebook, so retain it strictly as a
        # backwards-compatible fallback during a first deployment.
        try:
            cur.execute("SELECT * FROM forecast_nowcast ORDER BY FIELD(target_name, "
                        "'revenue', 'billings', 'rpo', 'customers_100k')")
            nowcast = cur.fetchall()
        except pymysql.MySQLError:
            cur.execute("SELECT * FROM forecast_latest ORDER BY as_of DESC")
            nowcast = cur.fetchall()
        competitors = {"npm": [], "revenue": []}
        try:
            cur.execute("SELECT `date`, company, package, value FROM competitor_npm_daily "
                        "WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) ORDER BY `date`, company")
            competitors["npm"] = cur.fetchall()
            cur.execute("SELECT company, quarter, period_end, revenue FROM competitor_revenue_quarterly "
                        "ORDER BY period_end, company")
            competitors["revenue"] = cur.fetchall()
        except Exception:
            pass
        fine = {}
        for key, query in {
            "product_daily": "SELECT * FROM product_downloads_fine_daily "
                             "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) "
                             "ORDER BY date, product_family, package",
            "product_monthly": "SELECT * FROM product_downloads_fine_monthly ORDER BY period, product_family, package",
            "product_quarterly": "SELECT * FROM product_downloads_fine_quarterly ORDER BY quarter, product_family, package",
            "hiring_family_quarterly": "SELECT * FROM hiring_lca_fine_by_family_quarterly ORDER BY quarter, job_family",
            "hiring_state_quarterly": "SELECT * FROM hiring_lca_fine_by_state_quarterly ORDER BY quarter, state",
        }.items():
            try:
                cur.execute(query)
                fine[key] = cur.fetchall()
            except Exception:
                fine[key] = []
        forum = []
        try:
            cur.execute("SELECT * FROM forum_mentions_daily "
                        "WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 730 DAY) ORDER BY `date`, source")
            forum = cur.fetchall()
        except Exception:
            pass

    rows = []
    for row in product:
        series = row["series_id"]
        source = "npm" if series.startswith("s1.npm.") else "pypi" if series.startswith("s1.pypi.") else None
        if source:
            package = series.split(".", 2)[2].removesuffix(".downloads")
            rows.append({"date": str(row["date"]), "source": source,
                         "metric": f"dl_{package}", "value": number(row["value"])})
    target_metrics = {"revenue": "revenue_q", "customers_100k": "customers_100k_arr",
                      "rpo": "rpo", "billings": "billings_q",
                      "free_cash_flow": "free_cash_flow_q", "revenue_yoy_pct": "revenue_yoy_pct"}
    for row in targets:
        for column, metric in target_metrics.items():
            value = number(row.get(column))
            if value is not None:
                rows.append({"date": str(row["period_end"]), "source": "sec_xbrl",
                             "metric": metric, "value": value})

    latest_hiring = max((str(x["date"]) for x in hiring_daily), default="—")
    totals = [number(x["value"]) for x in hiring_daily
              if x["series_id"] == "s2.greenhouse.total_open_roles" and str(x["date"]) == latest_hiring]
    by_department, by_geography = defaultdict(int), defaultdict(int)
    for job in jobs:
        if job.get("function"):
            by_department[job["function"]] += 1
        if job.get("geography"):
            by_geography[job["geography"]] += 1

    cloud_by_quarter = defaultdict(lambda: {"revenue": 0.0, "capex": 0.0, "companies": set()})
    for row in industry:
        bucket = cloud_by_quarter[row["quarter"]]
        bucket["revenue"] += number(row["revenue"]) or 0
        bucket["capex"] += number(row["capex"]) or 0
        bucket["companies"].add(row["company"])
    cloud_rows = []
    for quarter in sorted(cloud_by_quarter):
        value = cloud_by_quarter[quarter]
        if len(value["companies"]) != 3:
            continue
        previous = cloud_by_quarter.get(f"{int(quarter[:4]) - 1}{quarter[4:]}")
        revenue_yoy = ((value["revenue"] / previous["revenue"] - 1) * 100
                       if previous and previous["revenue"] else None)
        capex_yoy = ((value["capex"] / previous["capex"] - 1) * 100
                     if previous and previous["capex"] else None)
        cloud_rows.append({"quarter": quarter, "revenue_sum": value["revenue"],
                           "capex_sum": value["capex"], "revenue_yoy_pct": revenue_yoy,
                           "capex_yoy_pct": capex_yoy})
    cloud_company_revenue = [
        {"quarter": row["quarter"], "company": row["company"], "revenue": number(row["revenue"])}
        for row in industry
        if row.get("company") in {"AMZN", "MSFT", "GOOGL"} and number(row.get("revenue")) is not None
    ]

    industry_weekly = defaultdict(float)
    for row in industry_activity:
        dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
        week = (dt - timedelta(days=dt.weekday())).date().isoformat()
        industry_weekly[(week, row["category"])] += number(row["value"]) or 0

    show = {"GDPC1": "美国实际 GDP", "A191RL1Q225SBEA": "美国 GDP 增速", "NEWORDER": "制造业新订单",
            "CFNAI": "芝加哥联储景气", "INDPRO": "工业生产", "IPG3344S": "计算机电子产出",
            "DFII10": "10年期实际利率", "DGS10": "10年期国债收益率",
            "FEDFUNDS": "联邦基金利率", "T10Y2Y": "收益率曲线", "NFCI": "金融条件指数",
            "CPIAUCSL": "CPI", "DTWEXBGS": "美元指数", "PAYEMS": "非农就业",
            "GACDFSA066MSFRBPHI": "费城联储地区景气",
            "GACDISA066MSFRBNY": "纽约联储地区景气",
            "WPU38110101": "托管与 IT 基础设施服务 PPI"}
    quarterly = {}
    for row in macro:
        sid = row["series_id"].removeprefix("macro.fred.")
        if sid not in show:
            continue
        date = str(row["date"])
        quarter = f"{date[:4]}Q{(int(date[5:7]) - 1) // 3 + 1}"
        key = (sid, quarter)
        if key not in quarterly or date > quarterly[key][0]:
            quarterly[key] = (date, number(row["value"]))
    macro_rows = []
    for sid in show:
        points = [(quarter, value[1]) for (key_sid, quarter), value in quarterly.items()
                  if key_sid == sid and value[1] is not None]
        values = [point[1] for point in points]
        mean = sum(values) / len(values) if values else 0
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if values else 0
        for quarter, value in sorted(points):
            macro_rows.append({"quarter": quarter, "series": sid, "value": value,
                               "display_raw": value, "z": round((value - mean) / sd, 3) if sd else 0})

    latest_target = next((row for row in reversed(targets) if number(row.get("revenue")) is not None), {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
        "hiring": {"as_of": latest_hiring, "open_roles": int(totals[-1]) if totals else len(jobs),
                   "by_department": dict(by_department), "by_geography": dict(by_geography),
                   "model_eligible": False},
        "kpi": {"quarter": latest_target.get("quarter"), "revenue_usd": number(latest_target.get("revenue")),
                "revenue_yoy_pct": number(latest_target.get("revenue_yoy_pct")),
                "customers_100k_arr": number(latest_target.get("customers_100k"))},
        "cloud": {"rows": cloud_rows, "company_revenue": cloud_company_revenue},
        "industry": {"weekly": [{"week": week, "category": category, "value": value}
                                  for (week, category), value in sorted(industry_weekly.items())]},
        "macro": {"rows": macro_rows, "labels": show,
                  "modes": {sid: "level" for sid in show}},
        "nowcast": nowcast,
        "competitors": competitors,
        "fine": fine,
        "forum": {"rows": forum,
                  "google_trends": {"available": False,
                                    "reason": "等待 Google Trends 官方 API 访问权限"}},
        "model": {}, "leadlag": [], "sources": [],
    }


def forum_details(limit=120):
    """Return a bounded, link-first view of public developer discussions."""
    with database() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT `date`, source, title, text, url, incident, evaluation, adoption, pricing "
            "FROM forum_mentions_detail "
            "WHERE COALESCE(title, '') <> '' OR COALESCE(text, '') <> '' "
            "ORDER BY `date` DESC, record_id DESC LIMIT %s",
            (max(1, min(int(limit), 200)),),
        )
        return cur.fetchall()


def quarter_of(day):
    day = str(day)
    return f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}"


def solve(matrix, vector):
    """Gauss-Jordan solver for the small ridge-regression system."""
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-10:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            scale = augmented[row][col]
            augmented[row] = [a - scale * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def normalise_forecast_factors(factors):
    return [factor for factor in factors if factor in FORECAST_FACTORS]


def forecast_cache_key(factors, target_name, target_lags):
    return f"v1:{target_name}:lag{target_lags}:" + ",".join(sorted(factors))


def is_default_forecast_request(factors, target_lags):
    return target_lags == 0 and set(factors) == set(FORECAST_DEFAULT_FACTORS) and len(factors) == len(FORECAST_DEFAULT_FACTORS)


def cached_forecast_payload(factors, target_name, target_lags):
    """Return a daily precomputed result only for the dashboard default model."""
    if not is_default_forecast_request(factors, target_lags):
        return None
    try:
        with database() as conn, conn.cursor() as cur:
            cur.execute("SELECT payload_json, computed_at FROM forecast_cache WHERE cache_key = %s",
                        (forecast_cache_key(factors, target_name, target_lags),))
            row = cur.fetchone()
    except pymysql.MySQLError:
        return None
    if not row:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, ValueError):
        return None
    payload["precomputed_at"] = str(row["computed_at"])
    return payload


def forecast_payload(factors, target_name="revenue", target_lags=0, *, use_cache=True):
    factors = normalise_forecast_factors(factors)
    if not factors:
        factors = ["npm_total"]
    target_lags = max(0, min(4, int(target_lags or 0)))
    if use_cache:
        cached = cached_forecast_payload(factors, target_name, target_lags)
        if cached is not None:
            return cached
    target = FORECAST_TARGETS.get(target_name, FORECAST_TARGETS["revenue"])
    with database() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT quarter, `{target['column']}` AS target_value FROM targets_quarterly "
                    f"WHERE `{target['column']}` IS NOT NULL ORDER BY quarter")
        targets = cur.fetchall()
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
        peer_npm = cur.fetchall()

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
    for row in peer_npm:
        peer_totals[(quarter_of(row["date"]), str(row["company"]).lower())] += number(row["value"]) or 0
    for (quarter, company), value in peer_totals.items():
        key = {"datadog": "peer_ddog_sdk", "dynatrace": "peer_dynatrace_sdk", "elastic": "peer_elastic_sdk"}.get(company)
        if key:
            features[quarter][key] = value

    target_rows = [(row["quarter"], number(row["target_value"])) for row in targets]
    target_rows = [(quarter, value) for quarter, value in target_rows if value is not None]
    observations = []
    model_features = factors + [f"target_lag_{lag}" for lag in range(1, target_lags + 1)]
    for index, (quarter, target_value) in enumerate(target_rows):
        values = features.get(quarter, {})
        lagged_values = [target_rows[index - lag][1] for lag in range(1, target_lags + 1)
                         if index >= lag]
        if (all(values.get(name) is not None for name in factors)
                and len(lagged_values) == target_lags):
            scale = 1e6 if target["unit"] == "usd" else 1
            observations.append((quarter, target_value / scale,
                                 [values[name] for name in factors] +
                                 [value / scale for value in lagged_values]))
    predictions = []
    minimum_train = max(8, len(model_features) + 4)
    for index in range(minimum_train, len(observations)):
        train = observations[:index]
        means = [sum(row[2][j] for row in train) / len(train) for j in range(len(model_features))]
        scales = [math.sqrt(sum((row[2][j] - means[j]) ** 2 for row in train) / len(train)) or 1
                  for j in range(len(model_features))]
        x = [[1.0] + [(row[2][j] - means[j]) / scales[j]
                      for j in range(len(model_features))] for row in train]
        y = [row[1] for row in train]
        size = len(model_features) + 1
        gram = [[sum(row[i] * row[j] for row in x) + (0.15 if i == j and i else 0)
                 for j in range(size)] for i in range(size)]
        rhs = [sum(row[i] * value for row, value in zip(x, y)) for i in range(size)]
        beta = solve(gram, rhs)
        if not beta:
            continue
        quarter, actual, values = observations[index]
        point = [1.0] + [(values[j] - means[j]) / scales[j]
                         for j in range(len(model_features))]
        predicted = sum(a * b for a, b in zip(point, beta))
        predicted = max(0, predicted) * scale
        predictions.append({"quarter": quarter, "actual": actual * scale,
                            "predicted": predicted, "residual": actual * scale - predicted})
    errors = [row["actual"] - row["predicted"] for row in predictions]
    metrics = {}
    if errors:
        metrics = {"n": len(errors), "mae": sum(abs(x) for x in errors) / len(errors),
                   "rmse": math.sqrt(sum(x * x for x in errors) / len(errors)),
                   "mape_pct": sum(abs(x) / row["actual"] for x, row in zip(errors, predictions)) / len(errors) * 100}
        if len(predictions) > 1:
            hits = sum((row["actual"] - previous["actual"]) *
                       (row["predicted"] - previous["actual"]) >= 0
                       for previous, row in zip(predictions, predictions[1:]))
            metrics["direction_hit_pct"] = hits / (len(predictions) - 1) * 100
    # Use exactly the factors and AR lags requested by the user to estimate
    # the current quarter.  Unlike the daily default cards, this never drops
    # unavailable selected factors silently.
    current_quarter = max(features) if features else None
    current_forecast = {"available": False, "quarter": current_quarter,
                        "financial_history_through": target_rows[-1][0] if target_rows else None}
    current_values = features.get(current_quarter, {}) if current_quarter else {}
    missing_current = [factor for factor in factors if current_values.get(factor) is None]
    minimum_train = max(8, len(model_features) + 4)
    if missing_current:
        current_forecast["reason"] = "Current-quarter values are unavailable for: " + ", ".join(missing_current)
    elif len(observations) < minimum_train:
        current_forecast["reason"] = "Insufficient complete historical quarters for this configuration"
    elif len(target_rows) < target_lags:
        current_forecast["reason"] = "Insufficient target history for the selected autoregressive lags"
    else:
        means = [sum(row[2][j] for row in observations) / len(observations)
                 for j in range(len(model_features))]
        scales = [math.sqrt(sum((row[2][j] - means[j]) ** 2 for row in observations) / len(observations)) or 1
                  for j in range(len(model_features))]
        design = [[1.0] + [(row[2][j] - means[j]) / scales[j]
                           for j in range(len(model_features))] for row in observations]
        response = [row[1] for row in observations]
        size = len(model_features) + 1
        gram = [[sum(row[i] * row[j] for row in design) + (0.15 if i == j and i else 0)
                 for j in range(size)] for i in range(size)]
        beta = solve(gram, [sum(row[i] * value for row, value in zip(design, response))
                            for i in range(size)])
        if not beta:
            current_forecast["reason"] = "Unable to fit the selected configuration"
        else:
            lag_values = [target_rows[-lag][1] / scale for lag in range(1, target_lags + 1)]
            point = [1.0] + [(current_values[factor] - means[index]) / scales[index]
                             for index, factor in enumerate(factors)]
            point += [(value - means[len(factors) + index]) / scales[len(factors) + index]
                      for index, value in enumerate(lag_values)]
            current_forecast.update({
                "available": True,
                "forecast_value": max(0, sum(weight * value for weight, value in zip(beta, point))) * scale,
                "signals_through": max((str(row["date"]) for row in product + macro + peer_npm), default=None),
            })
    # Correlation is a shared diagnostic across the complete candidate universe.
    # It must not change just because one target model uses a smaller subset.
    correlation_factors = list(FORECAST_FACTORS)
    complete = [features[quarter] for quarter in sorted(features)
                if all(features[quarter].get(name) is not None for name in correlation_factors)]
    matrix = []
    for left in correlation_factors:
        row = []
        left_values = [float(values[left]) for values in complete]
        left_mean = sum(left_values) / len(left_values) if left_values else 0
        left_sd = math.sqrt(sum((value - left_mean) ** 2 for value in left_values))
        for right in correlation_factors:
            right_values = [float(values[right]) for values in complete]
            right_mean = sum(right_values) / len(right_values) if right_values else 0
            right_sd = math.sqrt(sum((value - right_mean) ** 2 for value in right_values))
            covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values))
            row.append(covariance / (left_sd * right_sd) if left_sd and right_sd else 0)
        matrix.append(row)
    return {"target": target["label"], "target_name": target_name, "unit": target["unit"], "factors": factors,
            "target_lags": target_lags,
            "factor_labels": FORECAST_FACTORS,
            "predictions": predictions, "metrics": metrics,
            "current_forecast": current_forecast,
            "available_factors": FORECAST_FACTORS, "factor_groups": FORECAST_FACTOR_GROUPS,
            "correlation": {"factors": correlation_factors, "matrix": matrix},
            "target_diagnostics": target_series_diagnostics(targets)}


class Handler(BaseHTTPRequestHandler):
    def json(self, status, body):
        payload = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        if use_gzip:
            payload = gzip.compress(payload, compresslevel=6)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        path, _, query = self.path.partition("?")
        if path.rstrip("/") == "/tables":
            return self.json(HTTPStatus.OK, {"tables": [{"name": name, "group": group}
                                                      for name, group in TABLES.items()]})
        if path.rstrip("/") == "/table":
            params = dict(item.split("=", 1) if "=" in item else (item, "")
                          for item in query.split("&") if item)
            name = params.get("name", "")
            if name not in TABLES:
                return self.json(HTTPStatus.BAD_REQUEST, {"error": "unknown table"})
            try:
                limit = max(1, min(int(params.get("limit", "200")), 1000))
            except ValueError:
                limit = 200
            try:
                with database() as conn, conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM `{name}` LIMIT %s", (limit,))
                    rows = cur.fetchall()
                return self.json(HTTPStatus.OK, {"name": name, "group": TABLES[name],
                                                  "rows": rows, "limit": limit})
            except Exception as exc:  # pragma: no cover - server boundary
                return self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        if path.rstrip("/") == "/forecast":
            try:
                params = parse_qs(query)
                factors = params.get("factors", [""])[0].split(",")
                return self.json(HTTPStatus.OK, forecast_payload(
                    factors, params.get("target", ["revenue"])[0],
                    params.get("target_lags", [0])[0]
                ))
            except Exception as exc:  # pragma: no cover - server boundary
                return self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        if path.rstrip("/") == "/forum-details":
            try:
                params = parse_qs(query)
                limit = params.get("limit", ["120"])[0]
                return self.json(HTTPStatus.OK, {"rows": forum_details(limit)})
            except Exception as exc:  # noqa: BLE001
                return self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})

        if self.path.rstrip("/") == "/payload":
            try:
                return self.json(HTTPStatus.OK, live_payload())
            except Exception as exc:  # pragma: no cover - server boundary
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if self.path.rstrip("/") == "/health":
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, _format, *_args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8091), Handler).serve_forever()
