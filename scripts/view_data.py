"""数据查看工具 —— 用一条命令把当前采集到的数据全部摊开给你看。

用法：
    .venv/bin/python scripts/view_data.py                 # 总览：文件清单 + 数据字典 + 最新值
    .venv/bin/python scripts/view_data.py --series npm    # 只看序列名含 "npm" 的数据
    .venv/bin/python scripts/view_data.py --tail 20       # 每个序列显示最近 20 行
    .venv/bin/python scripts/view_data.py --raw           # 同时列出原始快照文件

如果你更习惯用图形界面，直接双击打开 data/processed/*.csv（Excel/Numbers 可直接读，
CSV 用 utf-8-sig 编码，中文不会乱码）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, RAW_DIR, ROOT  # noqa: E402

def human(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def inventory(raw: bool = False) -> None:
    print("=" * 100)
    print("① 数据文件清单")
    print("=" * 100)
    rows = []
    for p in sorted(PROCESSED_DIR.glob("*")):
        if p.suffix not in (".csv", ".parquet"):
            continue
        n = ""
        if p.suffix == ".csv":
            try:
                n = f"{len(pd.read_csv(p)):,}"
            except Exception:
                n = "?"
        rows.append({"文件": str(p.relative_to(ROOT)), "类型": p.suffix.lstrip("."),
                     "行数": n, "大小": human(p.stat().st_size)})
    print(pd.DataFrame(rows).to_string(index=False))

    if raw:
        raw_files = sorted(RAW_DIR.rglob("*.json"))
        print(f"\n原始快照：{len(raw_files)} 个文件，位于 {RAW_DIR.relative_to(ROOT)}/")
        for p in raw_files[:10]:
            print(f"   {p.relative_to(ROOT)}  ({human(p.stat().st_size)})")
        if len(raw_files) > 10:
            print(f"   … 另有 {len(raw_files) - 10} 个")


def dictionary(flt: str | None, tail: int) -> None:
    dd_path = PROCESSED_DIR / "data_dictionary.csv"
    if not dd_path.exists():
        print("! 还没有 data_dictionary.csv，请先运行 scripts/build_dataset.py")
        return
    dd = pd.read_csv(dd_path)
    if flt:
        dd = dd[dd["series_id"].str.contains(flt, case=False, na=False)]
    print("\n" + "=" * 100)
    print("② 数据字典（已采集的每一个序列）")
    print("=" * 100)
    show = dd[["series_id", "rows", "first_date", "last_date", "latest_value", "unit"]].copy()
    show["latest_value"] = show["latest_value"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    print(show.to_string(index=False))
    print(f"\n共 {len(dd)} 个序列")

    obs_path = PROCESSED_DIR / "observations.csv"
    if not obs_path.exists():
        return
    obs = pd.read_csv(obs_path)
    if flt:
        obs = obs[obs["series_id"].str.contains(flt, case=False, na=False)]

    print("\n" + "=" * 100)
    print(f"③ 每个序列最近 {tail} 行")
    print("=" * 100)
    for sid, g in obs.groupby("series_id"):
        g = g.sort_values("date")
        print(f"\n--- {sid}   （{len(g):,} 行，{g['date'].iloc[0]} → {g['date'].iloc[-1]}，单位 {g['unit'].iloc[0]}）")
        print(g.tail(tail)[["date", "value"]].to_string(index=False))


def target() -> None:
    p = PROCESSED_DIR / "ddog_target_quarterly.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    print("\n" + "=" * 100)
    print("④ 目标变量：DDOG 已披露季度 KPI（预测的 y）")
    print("=" * 100)
    cols = [c for c in ["quarter", "revenue", "revenue_yoy_pct", "billings", "rpo",
                        "customers_100k", "customers_100k_yoy_pct", "free_cash_flow"] if c in df.columns]
    d = df[cols].tail(14).copy()
    for c in ["revenue", "billings", "rpo", "free_cash_flow"]:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v/1e6:,.0f}M" if pd.notna(v) else "")
    if "customers_100k" in d:
        d["customers_100k"] = d["customers_100k"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    for c in ["revenue_yoy_pct", "customers_100k_yoy_pct"]:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
    print(d.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", help="按序列名过滤，如 npm / hiring / pypi")
    ap.add_argument("--tail", type=int, default=5, help="每个序列显示最近几行（默认 5）")
    ap.add_argument("--raw", action="store_true", help="同时列出原始快照文件")
    a = ap.parse_args()

    inventory(raw=a.raw)
    dictionary(a.series, a.tail)
    if not a.series:
        target()

    print("\n" + "=" * 100)
    print("怎么看这些数据")
    print("=" * 100)
    print(f"""  1) 图形界面：双击打开 {PROCESSED_DIR.relative_to(ROOT)}/ 下的 .csv（Excel/Numbers 直接读）
  2) 命令行  ：.venv/bin/python scripts/view_data.py --series npm --tail 10
  3) Pandas  ：
       import pandas as pd
       df = pd.read_parquet("data/processed/observations.parquet")   # 规范长表
       df.pivot_table(index="date", columns="series_id", values="value")  # 转宽表
  4) 图表    ：.venv/bin/python scripts/make_charts.py  →  reports/figures/*.png
  5) 原始快照：data/raw/<source>/<日期>/*.json（含抓取 URL、时间、合规依据，可审计）""")


if __name__ == "__main__":
    main()
