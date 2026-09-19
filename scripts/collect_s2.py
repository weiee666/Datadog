"""S2 · Datadog 自身招聘速度采集器。

数据源：Greenhouse Job Board API（官方文档化的公开接口，专为职位分发设计）。
        **不抓 careers.datadoghq.com**（Datadog AUP 禁止）。

Greenhouse 返回的岗位不含 departments 字段，但带一组高质量自定义字段（实测）：
    Cost Center        真实组织单元，如 "Sales - Channels & Alliances"、"Engineering - Dev Eng"
    Geography          North America / EMEA / APAC / LATAM
    Area - Engineering Backend / Data Science / AI Engineering / ...
    IC or MG           Individual Contributor 还是 Manager
这些比笼统的"部门"更有信息量：可以构造「销售/渠道扩张」「AI 投入」「存量支持」等结构性前瞻信号。

产出：
  data/processed/s2_jobs_latest.csv        当前在招岗位明细（含全部自定义字段）
  data/processed/s2_hiring_daily.csv       长表：按日 × 各维度岗位数（逐日累积）
  data/processed/s2_hiring_daily_wide.csv  宽表：date + 职能列（Excel 直读）
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import LEGAL_BASIS, PROCESSED_DIR, get_json, save_raw, write_table  # noqa: E402

BOARD = "datadog"
JOBS_URL = f"https://boards-api.greenhouse.io/v1/boards/{BOARD}/jobs?content=false"

# 需要单独跟踪的 Cost Center（结构性前瞻信号）
WATCH_COST_CENTERS = [
    "Sales - Enterprise Sales",
    "Sales - Commercial Sales",
    "Sales - Sales Development",
    "Sales - Channels & Alliances",
    "Technical Solutions - Enterprise Sales Engineering",
    "Technical Solutions - Support Engineering",
    "Engineering - Dev Eng",
]


def md(job: dict, name: str):
    for m in job.get("metadata") or []:
        if m.get("name") == name:
            return m.get("value")
    return None


def upsert(path: Path, new: pd.DataFrame, key: str = "date") -> pd.DataFrame:
    """按日期 upsert：保留历史累积的日频序列，重复运行只覆盖当天。"""
    if path.exists():
        old = pd.read_csv(path)
        old = old[old[key].astype(str) != str(new[key].iloc[0])]
        return pd.concat([old, new], ignore_index=True).sort_values(key)
    return new


def main() -> None:
    print("== S2 采集开始（Greenhouse 公开 Job Board API） ==")
    payload = get_json(JOBS_URL, source="greenhouse", note="Datadog job board (public API)")
    jobs = payload.get("jobs", [])
    save_raw("greenhouse", "jobs", payload, url=JOBS_URL, note=f"Datadog board, {len(jobs)} jobs")
    today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    print(f"  在招岗位：{len(jobs)}（快照日期 {today}）")

    # ---------------- 明细表 ----------------
    rows = []
    for j in jobs:
        cc = md(j, "Cost Center") or ""
        rows.append(
            {
                "job_id": j.get("id"),
                "title": j.get("title"),
                "function": cc.split(" - ")[0].strip() if cc else "",
                "cost_center": cc,
                "area_engineering": md(j, "Area - Engineering") or "",
                "geography": md(j, "Geography") or "",
                "ic_or_mg": md(j, "IC or MG") or "",
                "time_type": md(j, "Time Type") or "",
                "location": (j.get("location") or {}).get("name", ""),
                "updated_at": (j.get("updated_at") or "")[:10],
                "first_published": (j.get("first_published") or "")[:10],
                "url": j.get("absolute_url"),
            }
        )
    detail = pd.DataFrame(rows).sort_values(["function", "cost_center", "title"])

    # ---------------- 按日聚合（长表） ----------------
    def obs(series: str, value: int):
        return {"date": today, "series_id": series, "value": value, "unit": "open roles",
                "freq": "D", "source": "greenhouse", "legal_basis": LEGAL_BASIS["greenhouse"]}

    records = [obs("s2.greenhouse.total_open_roles", len(detail))]
    for col, tag in [("function", "function"), ("geography", "geo"), ("ic_or_mg", "type")]:
        for k, n in Counter(detail[col]).most_common():
            if k:
                records.append(obs(f"s2.greenhouse.{tag}.{k}.open_roles", n))
    for k, n in Counter(detail["cost_center"]).most_common():
        if k in WATCH_COST_CENTERS:
            records.append(obs(f"s2.greenhouse.costcenter.{k}.open_roles", n))
    for k, n in Counter(detail["area_engineering"]).most_common():
        if k and k in ("AI Engineering", "Data Science", "Backend"):
            records.append(obs(f"s2.greenhouse.area.{k}.open_roles", n))

    daily_new = pd.DataFrame(records)
    daily = upsert(PROCESSED_DIR / "s2_hiring_daily.csv", daily_new)

    print("\n== 写出 ==")
    for p in write_table(detail, "s2_jobs_latest"):
        print("  ", p)
    for p in write_table(daily, "s2_hiring_daily"):
        print("  ", p)

    # ---------------- 宽表（职能为列） ----------------
    fn = daily[daily["series_id"].str.startswith("s2.greenhouse.function.")].copy()
    if len(fn):
        fn["cat"] = fn["series_id"].str.split(".").str[3]
        wide = fn.pivot_table(index="date", columns="cat", values="value", aggfunc="sum")
        wide.insert(0, "total_open_roles",
                    daily[daily["series_id"] == "s2.greenhouse.total_open_roles"]
                    .set_index("date")["value"])
        wide.index.name = "date"
        wide = wide.reset_index()
    else:
        wide = daily[daily["series_id"] == "s2.greenhouse.total_open_roles"].copy()
    for p in write_table(wide, "s2_hiring_daily_wide"):
        print("  ", p)

    print(f"\n完成：{len(detail)} 个岗位 / {detail['function'].nunique()} 个职能 / "
          f"{detail['geography'].nunique()} 个地区 / {detail['cost_center'].nunique()} 个 Cost Center")
    print("职能分布：", dict(Counter(detail["function"]).most_common(8)))
    print("注意：接口只给『当前快照』→ 日频序列从今天起逐日累积；")
    print("      历史回溯需接 DOL H-1B/LCA 季度数据（scripts/collect_dol.py，待建）。")


if __name__ == "__main__":
    main()
