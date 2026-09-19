"""Load the processed dashboard tables and its rendered payload into MySQL."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymysql

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TABLES = {
    "targets_quarterly": "ddog_target_quarterly.csv",
    "product_daily": "s1_instrumentation_daily.csv",
    "competitor_npm_daily": "competitor_npm_daily.csv",
    "competitor_revenue_quarterly": "competitor_revenue_quarterly.csv",
    "industry_activity_daily": "industry_activity_daily.csv",
    "industry_activity_cumulative": "industry_activity_cumulative_snapshots.csv",
    "hiring_daily": "s2_hiring_daily.csv",
    "hiring_jobs_snapshot": "s2_jobs_latest.csv",
    "headcount_annual": "s2_headcount_annual.csv",
    "macro_latest": "macro_factors_long.csv",
    "industry_quarterly": "cloud_complex_quarterly.csv",
    "forecast_latest": "forecast_latest.csv",
    "forecast_oos": "forecast_oos.csv",
    "forecast_metrics": "forecast_metrics.csv",
    "product_downloads_fine_daily": "product_downloads_fine_daily.csv",
    "product_downloads_fine_monthly": "product_downloads_fine_monthly.csv",
    "product_downloads_fine_quarterly": "product_downloads_fine_quarterly.csv",
    "hiring_lca_fine_detail": "hiring_lca_fine_detail.csv",
    "hiring_lca_fine_by_family_quarterly": "hiring_lca_fine_by_family_quarterly.csv",
    "hiring_lca_fine_by_state_quarterly": "hiring_lca_fine_by_state_quarterly.csv",
    "forum_mentions_daily": "forum_mentions_daily.csv",
    "forum_mentions_detail": "forum_mentions_detail.csv",
}


TABLEAU_VIEWS = {
    "tableau_targets": "SELECT * FROM targets_quarterly",
    "tableau_npm": """
        SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue
        FROM product_daily
        WHERE series_id = 's1.npm.dd-trace.downloads'
        GROUP BY DATE_FORMAT(`date`, '%Y-%m')
    """,
    "tableau_pypi": """
        SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue
        FROM product_daily
        WHERE series_id = 's1.pypi.ddtrace.downloads'
        GROUP BY DATE_FORMAT(`date`, '%Y-%m')
    """,
    "tableau_jobs": "SELECT `function` AS quarter, COUNT(*) AS revenue FROM hiring_jobs_snapshot GROUP BY `function`",
    "tableau_headcount": "SELECT CAST(fiscal_year AS CHAR) AS quarter, employees AS revenue FROM headcount_annual",
    "tableau_docker": "SELECT UTC_DATE() AS quarter, 11276023586 AS revenue",
    "tableau_terraform": "SELECT UTC_DATE() AS quarter, 546454342 AS revenue",
    "tableau_industry_revenue": """
        SELECT quarter, SUM(revenue) / 1000000000 AS revenue
        FROM industry_quarterly
        WHERE quarter >= '2023Q1'
        GROUP BY quarter HAVING COUNT(DISTINCT company) = 3
    """,
    "tableau_industry_capex": """
        SELECT quarter, SUM(capex) / 1000000000 AS revenue
        FROM industry_quarterly
        WHERE quarter >= '2023Q1'
        GROUP BY quarter HAVING COUNT(DISTINCT company) = 3
    """,
    "tableau_dgs10": "SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue FROM macro_latest WHERE series_id = 'macro.fred.DGS10' GROUP BY DATE_FORMAT(`date`, '%Y-%m')",
    "tableau_fedfunds": "SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue FROM macro_latest WHERE series_id = 'macro.fred.FEDFUNDS' GROUP BY DATE_FORMAT(`date`, '%Y-%m')",
    "tableau_nfci": "SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue FROM macro_latest WHERE series_id = 'macro.fred.NFCI' GROUP BY DATE_FORMAT(`date`, '%Y-%m')",
    "tableau_payems": "SELECT DATE_FORMAT(`date`, '%Y-%m') AS quarter, AVG(value) AS revenue FROM macro_latest WHERE series_id = 'macro.fred.PAYEMS' GROUP BY DATE_FORMAT(`date`, '%Y-%m')",
    # Dedicated views for the executive Tableau workbook.  These retain the
    # source-level fields instead of forcing every chart through `quarter` and
    # `revenue`, which keeps the workbook inspectable after refresh.
    "tableau_product_daily": "SELECT `date`, series_id, value, source FROM product_daily",
    "tableau_product_quarterly": "SELECT quarter, product_family, downloads FROM product_downloads_fine_quarterly",
    "tableau_competitor_npm": "SELECT `date`, company, package, value FROM competitor_npm_daily",
    "tableau_competitor_revenue": "SELECT company, quarter, period_end, revenue FROM competitor_revenue_quarterly",
    "tableau_industry_quarterly_detail": "SELECT company, quarter, period_end, revenue, capex FROM industry_quarterly",
    "tableau_macro_detail": "SELECT `date`, series_id, value, source FROM macro_latest WHERE series_id = 'macro.fred.DGS10'",
    "tableau_forecast_oos": "SELECT quarter, model AS target, y_true AS actual, y_pred AS prediction, y_pred - y_true AS residual FROM forecast_oos",
    "tableau_forecast_metrics": "SELECT * FROM forecast_metrics",
    "tableau_peer_sdk_ddog": "SELECT DATE_FORMAT(`date`, '%Y-%m-%d') AS quarter, value AS revenue FROM competitor_npm_daily WHERE company = 'Datadog'",
    "tableau_peer_sdk_dynatrace": "SELECT DATE_FORMAT(`date`, '%Y-%m-%d') AS quarter, value AS revenue FROM competitor_npm_daily WHERE company = 'Dynatrace'",
    "tableau_peer_sdk_elastic": "SELECT DATE_FORMAT(`date`, '%Y-%m-%d') AS quarter, value AS revenue FROM competitor_npm_daily WHERE company = 'Elastic'",
    "tableau_peer_revenue_ddog": "SELECT quarter, revenue FROM competitor_revenue_quarterly WHERE company = 'Datadog'",
    "tableau_peer_revenue_dynatrace": "SELECT quarter, revenue FROM competitor_revenue_quarterly WHERE company = 'Dynatrace'",
    "tableau_peer_revenue_elastic": "SELECT quarter, revenue FROM competitor_revenue_quarterly WHERE company = 'Elastic'",
    "tableau_forecast_actual": "SELECT quarter, AVG(y_true) AS revenue FROM forecast_oos WHERE model = '⑩ 全特征ridge' GROUP BY quarter",
    "tableau_forecast_predicted": "SELECT quarter, AVG(y_pred) AS revenue FROM forecast_oos WHERE model = '⑩ 全特征ridge' GROUP BY quarter",
    "tableau_forecast_residual": "SELECT quarter, AVG(y_pred - y_true) AS revenue FROM forecast_oos WHERE model = '⑩ 全特征ridge' GROUP BY quarter",
}


def connect():
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"], password=os.environ["MYSQL_PASSWORD"],
        database=os.environ.get("MYSQL_DATABASE", "ddog_dashboard"),
        charset="utf8mb4", autocommit=False,
    )


def identifier(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier: {name}")
    return f"`{name}`"


def sql_type(column: str) -> str:
    if column in {"date", "period", "period_end", "filing_date", "first_filed", "latest_filed", "as_of", "updated_at", "first_published"}:
        return "DATE NULL"
    if column.endswith("_pct") or column in {"value", "downloads", "revenue", "rpo", "billings", "capex", "revenue_sum", "capex_sum", "forecast_yoy_pct", "implied_revenue_usd", "new_employment", "continued_employment", "median_wage_annual", "wage_annual"}:
        return "DOUBLE NULL"
    if column in {"job_id", "employees", "countries", "customers_100k", "npm_days_observed", "active_days", "lca_cases", "certified_cases", "states"}:
        return "BIGINT NULL"
    return "TEXT NULL"


def load_table(cur, name: str, frame: pd.DataFrame) -> int:
    columns = list(frame.columns)
    definition = ", ".join(f"{identifier(c)} {sql_type(c)}" for c in columns)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {identifier(name)} ({definition}) CHARACTER SET utf8mb4")
    # Processed tables can gain provenance fields as collectors improve.  Keep
    # existing production tables intact and add only the newly required columns.
    cur.execute(f"SHOW COLUMNS FROM {identifier(name)}")
    existing = {row[0] for row in cur.fetchall()}
    for column in columns:
        if column not in existing:
            cur.execute(f"ALTER TABLE {identifier(name)} ADD COLUMN {identifier(column)} {sql_type(column)}")
    cur.execute(f"TRUNCATE TABLE {identifier(name)}")
    insert = f"INSERT INTO {identifier(name)} ({', '.join(identifier(c) for c in columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
    def database_value(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if value is None or pd.isna(value):
            return None
        return value

    values = [tuple(database_value(value) for value in row)
              for row in frame.itertuples(index=False, name=None)]
    if values:
        cur.executemany(insert, values)
    return len(values)


def create_tableau_views(cur):
    for name, query in TABLEAU_VIEWS.items():
        cur.execute(f"CREATE OR REPLACE VIEW {identifier(name)} AS {query}")


def main():
    payload = (ROOT / "site" / "data" / "payload.json").read_text(encoding="utf-8")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS dashboard_payload (id TINYINT PRIMARY KEY, payload_json LONGTEXT NOT NULL, updated_at DATETIME NOT NULL)")
        cur.execute("CREATE TABLE IF NOT EXISTS data_runs (id BIGINT AUTO_INCREMENT PRIMARY KEY, started_at DATETIME NOT NULL, finished_at DATETIME NULL, status VARCHAR(20) NOT NULL, details TEXT NULL)")
        cur.execute("INSERT INTO data_runs (started_at, status) VALUES (UTC_TIMESTAMP(), 'running')")
        run_id = cur.lastrowid
        counts = {}
        try:
            for table, filename in TABLES.items():
                frame = pd.read_csv(PROCESSED / filename)
                counts[table] = load_table(cur, table, frame)
            create_tableau_views(cur)
            cur.execute("REPLACE INTO dashboard_payload (id, payload_json, updated_at) VALUES (1, %s, UTC_TIMESTAMP())", (payload,))
            cur.execute("UPDATE data_runs SET finished_at = UTC_TIMESTAMP(), status = 'success', details = %s WHERE id = %s", (json.dumps(counts), run_id))
            conn.commit()
        except Exception as exc:
            conn.rollback()
            cur.execute("UPDATE data_runs SET finished_at = UTC_TIMESTAMP(), status = 'failed', details = %s WHERE id = %s", (str(exc), run_id))
            conn.commit()
            raise
    print("MySQL updated", counts, datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
