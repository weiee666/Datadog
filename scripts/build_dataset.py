"""把所有采集结果合并成一张规范长表（canonical tidy table）并生成数据字典。

规范表 data/processed/observations.csv 的列：
    date       观察日期 (YYYY-MM-DD)
    series_id  序列唯一标识，形如 s1.npm.dd-trace.downloads
    value      数值
    unit       单位
    freq       频率 D/W/Q
    source     数据来源
    legal_basis 合规依据（可直接放进报告的 Data Provenance 表）

同时生成：
    data/processed/data_dictionary.csv   全部序列的清单与统计
    data/processed/observations_wide.csv 仅日频信号，按日期透视成宽表（Excel 直读）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, write_table  # noqa: E402

SERIES_DESC = {
    "s1.": "S1 开发者插桩流量（用量原料）",
    "industry.": "行业活动因子 · 跨厂商开源可观测性与云基础设施生态",
    "s2.greenhouse": "S2 招聘 · Greenhouse 实时在招岗位（日频，逐日累积）",
    "s2.dol.": "S2 招聘 · DOL H-1B/LCA 季度披露（历史回溯）",
    "s2.headcount": "S2 人力 · 10-K 员工总数（年度锚点）",
    "target.": "目标变量（DDOG 已披露 KPI）",
    "macro.fred.": "宏观环境因子 · FRED/ALFRED 公共领域序列（GDP/工业生产/资本开支订单/利率/金融条件/通胀/美元）",
    "macro.wb.": "宏观环境因子 · 世界银行年度 GDP 增速（背景口径，不进季度模型）",
    "macro.cloud.": "行业景气因子 · 超大规模云厂商收入与资本开支（SEC XBRL）",
}


def _desc(series_id: str) -> str:
    for k, v in SERIES_DESC.items():
        if series_id.startswith(k):
            return v
    return ""


def headcount_long() -> pd.DataFrame:
    """把 10-K 年度人力表转成长表（date 锚在报告期末）。"""
    p = PROCESSED_DIR / "s2_headcount_annual.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "period_end" not in df.columns:
        df["period_end"] = df["fiscal_year"].astype(str) + "-12-31"
    spec = {"employees": ("people", "员工总数"),
            "employees_rnd": ("people", "研发员工数"),
            "countries": ("countries", "覆盖国家数")}
    recs = []
    for _, r in df.iterrows():
        for col, (unit, _d) in spec.items():
            if col in df.columns and pd.notna(r.get(col)):
                recs.append({"date": r["period_end"], "series_id": f"s2.headcount.{col}",
                             "value": float(r[col]), "unit": unit, "freq": "Y", "source": "sec",
                             "legal_basis": r.get("legal_basis", "")})
    return pd.DataFrame(recs)


def cloud_factor_long() -> pd.DataFrame:
    """超大规模云厂商（AMZN/MSFT/GOOGL）收入与资本开支 -> 长表（行业景气因子）。"""
    p = PROCESSED_DIR / "cloud_complex_panel.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    spec = {"revenue_sum": ("USD", "三大云厂商季度收入合计（行业需求代理）"),
            "capex_sum": ("USD", "三大云厂商季度资本开支合计（云产能投入）"),
            "revenue_yoy_pct": ("% yoy", "三大云厂商收入合计同比"),
            "capex_yoy_pct": ("% yoy", "三大云厂商资本开支合计同比")}
    recs = []
    for _, r in df.iterrows():
        for col, (unit, desc) in spec.items():
            if col in df.columns and pd.notna(r.get(col)):
                recs.append({"date": r["period_end"], "series_id": f"macro.cloud.{col}",
                             "value": float(r[col]), "unit": unit, "freq": "Q",
                             "source": "sec", "legal_basis": r.get("legal_basis", "")})
    return pd.DataFrame(recs)


def main() -> None:
    parts = []
    for f in ["s1_instrumentation_daily.csv", "s1_cumulative_snapshots.csv",
              "industry_activity_daily.csv", "industry_activity_cumulative_snapshots.csv",
              "s2_hiring_daily.csv", "s2_dol_lca_long.csv", "macro_factors_long.csv"]:
        p = PROCESSED_DIR / f
        if p.exists():
            parts.append(pd.read_csv(p))
            print(f"  + {f} ({len(parts[-1]):,} 行)")
        else:
            print(f"  ! 缺失：{f}（先运行对应 collect_*.py）")

    hc = headcount_long()
    if len(hc):
        parts.append(hc)
        print(f"  + s2_headcount_annual.csv -> 长表 ({len(hc)} 行)")

    cf = cloud_factor_long()
    if len(cf):
        parts.append(cf)
        print(f"  + cloud_complex_panel.csv -> 长表 ({len(cf)} 行)")

    if not parts:
        raise SystemExit("没有任何可合并的数据，请先运行采集脚本。")

    obs = pd.concat(parts, ignore_index=True)
    obs["date"] = pd.to_datetime(obs["date"]).dt.date
    obs = obs.drop_duplicates(["date", "series_id"], keep="last").sort_values(["series_id", "date"])
    obs["description"] = obs["series_id"].map(_desc)

    for p in write_table(obs, "observations"):
        print("  ->", p)

    # 数据字典
    dd = (
        obs.groupby(["series_id", "unit", "freq", "source", "legal_basis", "description"], dropna=False)
        .agg(rows=("value", "size"), first_date=("date", "min"), last_date=("date", "max"),
             latest_value=("value", "last"), min_value=("value", "min"), max_value=("value", "max"))
        .reset_index()
        .sort_values("series_id")
    )
    for p in write_table(dd, "data_dictionary"):
        print("  ->", p)

    # 日频宽表
    daily = obs[obs["freq"] == "D"].copy()
    if len(daily):
        wide = daily.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
        wide.index.name = "date"
        wide = wide.reset_index()
        for p in write_table(wide, "observations_wide"):
            print("  ->", p)

    print(f"\n合计 {len(obs):,} 行，{obs['series_id'].nunique()} 个序列")


if __name__ == "__main__":
    main()
