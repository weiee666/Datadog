"""Build fine-grained product and hiring factors for MySQL-only analysis.

These tables are intentionally not wired into the static dashboard yet. They
answer the question: which detailed dimensions can be collected back to 2018+
at monthly or quarterly frequency?
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import LEGAL_BASIS, PROCESSED_DIR, RAW_DIR, write_table  # noqa: E402


PRODUCT_TAXONOMY = {
    # Only keep packages with a defensible mapping to one product family.  The
    # dashboard must not turn generic API, mobile, or logging package traffic
    # into a made-up product adoption signal.
    "s1.npm.dd-trace.downloads": ("APM", "Node.js tracing", "dd-trace"),
    "s1.npm.@datadog/native-metrics.downloads": ("APM", "native metrics", "@datadog/native-metrics"),
    "s1.npm.@datadog/libdatadog.downloads": ("APM", "libdatadog bindings", "@datadog/libdatadog"),
    "s1.npm.@datadog/native-appsec.downloads": ("Security", "AppSec native bindings", "@datadog/native-appsec"),
    "s1.npm.datadog-lambda-js.downloads": ("Serverless", "AWS Lambda JS", "datadog-lambda-js"),
    "s1.npm.serverless-plugin-datadog.downloads": ("Serverless", "Serverless plugin", "serverless-plugin-datadog"),
    "s1.npm.datadog-cdk-constructs-v2.downloads": ("Serverless", "AWS CDK constructs", "datadog-cdk-constructs-v2"),
    "s1.npm.@datadog/datadog-ci-plugin-lambda.downloads": ("Serverless", "CI Lambda plugin", "@datadog/datadog-ci-plugin-lambda"),
    "s1.npm.@datadog/browser-rum.downloads": ("Real User Monitoring", "Browser RUM", "@datadog/browser-rum"),
    "s1.npm.@datadog/browser-rum-core.downloads": ("Real User Monitoring", "Browser RUM core", "@datadog/browser-rum-core"),
    "s1.npm.@datadog/browser-rum-react.downloads": ("Real User Monitoring", "React RUM", "@datadog/browser-rum-react"),
    # The package is a CI runner, not a direct end-user Synthetic usage count.
    # It is kept as an explicitly labelled developer-adoption proxy only.
    "s1.npm.@datadog/datadog-ci.downloads": ("Synthetic Monitoring", "CI testing CLI (proxy)", "@datadog/datadog-ci"),
}

JOB_FAMILY_RULES = [
    ("AI / data", re.compile(r"ai|machine learning|data scien|research scientist", re.I)),
    ("Security", re.compile(r"security|appsec|threat|vulnerab", re.I)),
    ("Infrastructure / platform", re.compile(r"site reliability|sre|infrastructure|platform|cloud|devops", re.I)),
    ("Software engineering", re.compile(r"software|engineer|developer|backend|frontend|full stack|architect", re.I)),
    ("Product / design", re.compile(r"product|design|ux|user experience", re.I)),
    ("Sales / GTM", re.compile(r"sales|account|business development|solutions engineer|customer success", re.I)),
    ("G&A", re.compile(r"finance|legal|people|hr|recruit|operations|program manager", re.I)),
]


def month_end(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp("M").dt.strftime("%Y-%m-%d")


def quarter_label(s: pd.Series) -> pd.Series:
    p = s.dt.to_period("Q")
    return p.astype(str)


def quarter_end(s: pd.Series) -> pd.Series:
    return s.dt.to_period("Q").dt.to_timestamp("Q").dt.strftime("%Y-%m-%d")


def classify_job(title: str) -> str:
    title = title or ""
    for name, rx in JOB_FAMILY_RULES:
        if rx.search(title):
            return name
    return "Other / mixed"


def build_product_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = PROCESSED_DIR / "s1_instrumentation_daily.csv"
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d["series_id"].isin(PRODUCT_TAXONOMY)].copy()
    if not len(d):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] >= "2018-01-01"].copy()
    meta = d["series_id"].map(PRODUCT_TAXONOMY)
    d["product_family"] = meta.map(lambda x: x[0])
    d["product_signal"] = meta.map(lambda x: x[1])
    d["package"] = meta.map(lambda x: x[2])
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0)

    daily = d[["date", "series_id", "package", "product_family", "product_signal", "value", "unit", "source", "legal_basis"]].copy()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    monthly = (d.assign(period=month_end(d["date"]))
               .groupby(["period", "product_family", "product_signal", "package", "series_id"], as_index=False)
               .agg(downloads=("value", "sum"), active_days=("date", "nunique")))
    monthly["freq"] = "M"
    monthly["source"] = "npm"
    monthly["legal_basis"] = LEGAL_BASIS["npm"]

    quarterly = (d.assign(period=quarter_end(d["date"]), quarter=quarter_label(d["date"]))
                 .groupby(["quarter", "period", "product_family", "product_signal", "package", "series_id"], as_index=False)
                 .agg(downloads=("value", "sum"), active_days=("date", "nunique")))
    quarterly["freq"] = "Q"
    quarterly["source"] = "npm"
    quarterly["legal_basis"] = LEGAL_BASIS["npm"]
    return daily, monthly, quarterly


def load_dol_detail() -> pd.DataFrame:
    files = sorted((RAW_DIR / "dol").glob("FY*_Q*_datadog.csv"))
    frames = []
    for path in files:
        m = re.search(r"FY(\d+)_Q(\d+)_datadog", path.name)
        if not m:
            continue
        fy, fq = int(m.group(1)), int(m.group(2))
        q = f"{fy - 1}Q4" if fq == 1 else f"{fy}Q{fq - 1}"
        df = pd.read_csv(path)
        df["fiscal_year"] = fy
        df["fiscal_quarter"] = fq
        df["quarter"] = q
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_hiring_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = load_dol_detail()
    if not len(d):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d["period_end"] = pd.PeriodIndex(d["quarter"], freq="Q").to_timestamp("Q").strftime("%Y-%m-%d")
    d["job_title_clean"] = d["JOB_TITLE"].fillna("").astype(str).str.strip()
    d["job_family"] = d["job_title_clean"].map(classify_job)
    d["new_employment"] = pd.to_numeric(d.get("NEW_EMPLOYMENT"), errors="coerce").fillna(0)
    d["continued_employment"] = pd.to_numeric(d.get("CONTINUED_EMPLOYMENT"), errors="coerce").fillna(0)
    d["wage_annual"] = pd.to_numeric(d.get("WAGE_RATE_OF_PAY_FROM"), errors="coerce")
    d["state"] = d.get("WORKSITE_STATE", "").fillna("").astype(str)
    d["status"] = d.get("CASE_STATUS", "").fillna("").astype(str)

    detail_cols = ["quarter", "period_end", "job_family", "job_title_clean", "state", "status",
                   "new_employment", "continued_employment", "wage_annual", "SOC_CODE", "SOC_TITLE"]
    detail = d[[c for c in detail_cols if c in d.columns]].copy()
    detail["source"] = "dol"
    detail["legal_basis"] = LEGAL_BASIS["dol"]

    by_family = (d.groupby(["quarter", "period_end", "job_family"], as_index=False)
                 .agg(lca_cases=("CASE_NUMBER", "count"),
                      certified_cases=("status", lambda s: (s.str.lower() == "certified").sum()),
                      new_employment=("new_employment", "sum"),
                      continued_employment=("continued_employment", "sum"),
                      median_wage_annual=("wage_annual", "median"),
                      states=("state", "nunique")))
    by_family["freq"] = "Q"
    by_family["source"] = "dol"
    by_family["legal_basis"] = LEGAL_BASIS["dol"]

    by_state = (d.groupby(["quarter", "period_end", "state"], as_index=False)
                .agg(lca_cases=("CASE_NUMBER", "count"),
                     new_employment=("new_employment", "sum"),
                     median_wage_annual=("wage_annual", "median")))
    by_state["freq"] = "Q"
    by_state["source"] = "dol"
    by_state["legal_basis"] = LEGAL_BASIS["dol"]
    return detail, by_family, by_state


def main() -> None:
    print("== 细分产品/招聘因子 ==")
    product_daily, product_monthly, product_quarterly = build_product_tables()
    for df, name in [
        (product_daily, "product_downloads_fine_daily"),
        (product_monthly, "product_downloads_fine_monthly"),
        (product_quarterly, "product_downloads_fine_quarterly"),
    ]:
        if len(df):
            for p in write_table(df, name):
                print("  ", p)
        else:
            print(f"  ! {name}: no rows")

    hiring_detail, hiring_family, hiring_state = build_hiring_tables()
    for df, name in [
        (hiring_detail, "hiring_lca_fine_detail"),
        (hiring_family, "hiring_lca_fine_by_family_quarterly"),
        (hiring_state, "hiring_lca_fine_by_state_quarterly"),
    ]:
        if len(df):
            for p in write_table(df, name):
                print("  ", p)
        else:
            print(f"  ! {name}: no rows")

    print("完成：细分表已写入 data/processed，后续 sync_mysql.py 会同步到 MySQL。")


if __name__ == "__main__":
    main()
