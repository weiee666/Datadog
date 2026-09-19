"""S1 · 开发者插桩流量采集器（增量 upsert 版）。

数据源（全部为第三方公共 API，不触碰 Datadog 自有站点）：
  - npm registry downloads API   -> Node.js / Serverless / RUM / CI 的 SDK 下载量
  - pypistats.org                -> Python ddtrace 等包下载量
  - Terraform Registry API       -> DataDog provider 累计下载（新组织开通代理）
  - Docker Hub API               -> datadog/agent 累计拉取

增量逻辑：重复运行不会重复下载历史。每个序列从「已存数据的最后一天 - 30 天」开始
补抓并 upsert，因此可以安全地挂到每日定时任务上。

产出：
  data/raw/npm|pypistats|terraform|dockerhub/<date>/*.json   原始快照（可审计）
  data/processed/s1_instrumentation_daily.csv                长表（规范格式，增量累积）
  data/processed/s1_instrumentation_weekly.csv               宽表（Excel 直读）
  data/processed/s1_cumulative_snapshots.csv                 累计型快照（逐日累积）
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import (LEGAL_BASIS, PROCESSED_DIR, NotFound, get_json,  # noqa: E402
                         save_raw, write_table)

NPM_PACKAGES = [
    # APM / backend tracing
    "dd-trace",                    # Node.js APM tracer —— 核心插桩流量
    "@datadog/native-metrics",      # Node native metrics collector
    "@datadog/native-appsec",       # AppSec native bindings
    "@datadog/libdatadog",          # libdatadog Node/WASM bindings
    # Serverless / cloud workload instrumentation
    "datadog-lambda-js",           # AWS Lambda 插桩
    "serverless-plugin-datadog",    # Serverless Framework 插件
    "datadog-cdk-constructs-v2",    # AWS CDK 自动插桩
    "@datadog/datadog-ci-plugin-lambda",
    # Browser RUM / logs
    "@datadog/browser-rum",        # 浏览器 RUM（账户级采用）
    "@datadog/browser-rum-core",
    "@datadog/browser-rum-react",
    "@datadog/browser-logs",       # 浏览器日志
    # Mobile
    "@datadog/mobile-react-native",
    "@datadog/mobile-react-navigation",
    "@datadog/mobile-react-native-session-replay",
    # CI / DevOps
    "@datadog/datadog-ci",         # CI Visibility（注意是 scope 包名）
    "@datadog/datadog-api-client",
]
PYPI_PACKAGES = ["ddtrace", "datadog", "datadog-api-client"]
NPM_START = "2016-01-01"
CHUNK_DAYS = 400          # npm range 接口上限 18 个月，取 400 天保守值
OVERLAP_DAYS = 30         # 增量补抓的重叠窗口，容忍接口延迟回填
DAILY_CSV = PROCESSED_DIR / "s1_instrumentation_daily.csv"
CUM_CSV = PROCESSED_DIR / "s1_cumulative_snapshots.csv"


def _chunks(start: date, end: date, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + timedelta(days=1)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "series_id", "value", "unit", "freq", "source", "legal_basis"])
    return pd.read_csv(path)


def upsert(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if not len(new):
        return existing
    both = pd.concat([existing, new], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"]).dt.date
    return (both.drop_duplicates(["date", "series_id"], keep="last")
            .sort_values(["series_id", "date"]).reset_index(drop=True))


def _row(d: str, series: str, value, unit: str, source: str) -> dict:
    return {"date": d, "series_id": series, "value": value, "unit": unit, "freq": "D",
            "source": source, "legal_basis": LEGAL_BASIS[source]}


def collect_npm(existing: pd.DataFrame) -> pd.DataFrame:
    rows, today = [], datetime.now(timezone.utc).date()
    for pkg in NPM_PACKAGES:
        sid = f"s1.npm.{pkg}.downloads"
        have = existing[existing["series_id"] == sid]
        if len(have):
            start = pd.to_datetime(have["date"].max()).date() - timedelta(days=OVERLAP_DAYS)
            mode = f"增量补抓（已有至 {have['date'].max()}）"
        else:
            start, mode = datetime.strptime(NPM_START, "%Y-%m-%d").date(), "全量历史"
        print(f"  npm: {pkg}  [{mode}]")
        got, skipped = 0, 0
        for a, b in _chunks(start, today, CHUNK_DAYS):
            url = f"https://api.npmjs.org/downloads/range/{a}:{b}/{quote(pkg, safe='')}"
            try:
                data = get_json(url, source="npm", note=f"npm daily downloads {pkg} {a}..{b}")
            except NotFound:
                skipped += 1          # 包尚未发布的区间，正常跳过
                continue
            except Exception as e:
                print(f"    ! {a}..{b}: {str(e)[:70]}")
                continue
            days = data.get("downloads") or []
            if not days:
                continue
            save_raw("npm", f"{pkg.replace('/', '_').replace('@', '')}_{a}_{b}", data,
                     url=url, note=f"npm daily downloads {pkg}")
            rows += [_row(x["day"], sid, x["downloads"], "downloads/day", "npm") for x in days]
            got += len(days)
        print(f"    -> {got} 天" + (f"（{skipped} 个区间因包未发布而跳过）" if skipped else ""))
    return pd.DataFrame(rows)


def collect_pypi(existing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pkg in PYPI_PACKAGES:
        sid = f"s1.pypi.{pkg}.downloads"
        url = f"https://pypistats.org/api/packages/{pkg}/overall"
        print(f"  pypi: {pkg}（接口仅提供约 180 天窗口，历史长度是硬限制）")
        try:
            data = get_json(url, source="pypistats", note=f"pypistats daily {pkg}")
        except Exception as e:
            print(f"    ! {str(e)[:70]}")
            continue
        save_raw("pypistats", f"{pkg}_overall", data, url=url, note=f"pypistats {pkg}")
        n = len([d for d in data.get("data", []) if d.get("category") == "without_mirrors"])
        rows += [_row(d["date"], sid, d["downloads"], "downloads/day", "pypistats")
                 for d in data.get("data", []) if d.get("category") == "without_mirrors"]
        print(f"    -> {n} 天")
    return pd.DataFrame(rows)


def collect_cumulative() -> pd.DataFrame:
    """累计型指标：接口只给累计值，只能逐日快照后自行差分。"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []

    url = "https://registry.terraform.io/v1/providers/DataDog/datadog"
    data = get_json(url, source="terraform", note="DataDog provider total downloads")
    save_raw("terraform", "provider_datadog", data, url=url, note="terraform provider stats")
    rows.append(_row(today, "s1.terraform.datadog.cum_downloads", data.get("downloads"),
                     "cumulative downloads", "terraform"))
    print(f"  terraform: {data.get('downloads'):,} 累计下载")

    url = "https://hub.docker.com/v2/repositories/datadog/agent/"
    data = get_json(url, source="dockerhub", note="datadog/agent pull count")
    save_raw("dockerhub", "datadog_agent", data, url=url, note="dockerhub repo stats")
    rows.append(_row(today, "s1.dockerhub.datadog-agent.cum_pulls", data.get("pull_count"),
                     "cumulative pulls", "dockerhub"))
    print(f"  dockerhub: {data.get('pull_count'):,} 累计拉取")
    return pd.DataFrame(rows)


def main() -> None:
    print("== S1 采集开始 ==")
    existing = load_existing(DAILY_CSV)
    if len(existing):
        print(f"  已有 {len(existing):,} 行历史（{existing['series_id'].nunique()} 个序列），走增量模式")

    new = pd.concat([collect_npm(existing), collect_pypi(existing)], ignore_index=True)
    daily = upsert(existing, new)

    cum = upsert(load_existing(CUM_CSV), collect_cumulative())

    print("\n== 写出 ==")
    for p in write_table(daily, "s1_instrumentation_daily"):
        print("  ", p)

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time.dt.date
    d = d[d["series_id"].str.startswith("s1.npm.")]      # 只有 npm 有 10 年历史，单独做周表
    wk = d.groupby(["week", "series_id"])["value"].sum().unstack("series_id").round(0)
    wk.index.name = "week_start"
    for p in write_table(wk.reset_index(), "s1_instrumentation_weekly"):
        print("  ", p)

    for p in write_table(cum, "s1_cumulative_snapshots"):
        print("  ", p)

    print(f"\n完成：{len(daily):,} 行日频观测；{daily['date'].min()} → {daily['date'].max()}；"
          f"{daily['series_id'].nunique()} 个序列")


if __name__ == "__main__":
    main()
