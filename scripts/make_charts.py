"""生成图表（PNG），让人一眼看到信号长什么样。

产出 reports/figures/：
  01_npm_weekly_downloads.png   S1：5 个 SDK 的周下载量，10 年历史
  02_signal_vs_revenue.png      S1 季度插桩量 vs DDOG 季度收入（双轴）+ 两者同比
  03_hiring_breakdown.png       S2：在招岗位的职能/地区结构
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, ROOT  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
# 中文字体（默认 DejaVu Sans 无 CJK 字形，会渲染成方块）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig1_npm() -> None:
    obs = pd.read_parquet(PROCESSED_DIR / "observations.parquet")
    npm = obs[obs["series_id"].str.startswith("s1.npm.")].copy()
    if not len(npm):
        return
    npm["date"] = pd.to_datetime(npm["date"])
    npm["week"] = npm["date"].dt.to_period("W-SUN").dt.start_time
    w = npm.groupby(["week", "series_id"])["value"].sum().unstack("series_id")
    w.columns = [c.replace("s1.npm.", "").replace(".downloads", "") for c in w.columns]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in w.columns:
        ax.plot(w.index, w[c], lw=1.2, label=c)
    ax.set_yscale("log")
    ax.set_title("S1 · Datadog SDK weekly downloads (npm, log scale) — 2016 to now")
    ax.set_ylabel("downloads / week (log)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "01_npm_weekly_downloads.png")
    plt.close(fig)
    print("  -> 01_npm_weekly_downloads.png")


def fig2_vs_revenue() -> None:
    obs = pd.read_parquet(PROCESSED_DIR / "observations.parquet")
    tgt = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    npm = obs[obs["series_id"].str.startswith("s1.npm.")].copy()
    if not len(npm) or not len(tgt):
        return
    npm["date"] = pd.to_datetime(npm["date"])
    npm["quarter"] = npm["date"].dt.to_period("Q").astype(str)
    q = npm.groupby("quarter")["value"].sum().rename("npm_downloads").reset_index()
    t = tgt[["quarter", "revenue", "revenue_yoy_pct", "customers_100k"]].dropna(subset=["revenue"])
    m = t.merge(q, on="quarter", how="inner").sort_values("quarter")
    m["npm_yoy"] = m["npm_downloads"].pct_change(4) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax1.plot(m["quarter"], m["npm_downloads"] / 1e6, "o-", color="#632ca6", label="npm downloads (M)")
    ax1.set_ylabel("npm downloads (millions)", color="#632ca6")
    ax1.tick_params(axis="y", labelcolor="#632ca6")
    ax1b = ax1.twinx()
    ax1b.plot(m["quarter"], m["revenue"] / 1e6, "s--", color="#e8a33d", label="DDOG revenue ($M)")
    ax1b.set_ylabel("DDOG revenue ($M)", color="#e8a33d")
    ax1b.tick_params(axis="y", labelcolor="#e8a33d")
    ax1b.grid(False)
    ax1.set_title("S1 vs reported revenue (quarterly)")
    ax1.tick_params(axis="x", rotation=90)

    ax2.plot(m["quarter"], m["revenue_yoy_pct"], "s--", color="#e8a33d", label="DDOG revenue YoY %")
    ax2.plot(m["quarter"], m["npm_yoy"], "o-", color="#632ca6", label="npm downloads YoY %")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("YoY growth (%)")
    ax2.set_title("Growth rates: signal vs revenue")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(FIG / "02_signal_vs_revenue.png")
    plt.close(fig)
    print("  -> 02_signal_vs_revenue.png")


def fig3_hiring() -> None:
    p = PROCESSED_DIR / "s2_hiring_daily.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    snap = d[d["date"] == d["date"].max()]
    fn = snap[snap["series_id"].str.contains(r"\.function\.")].copy()
    geo = snap[snap["series_id"].str.contains(r"\.geo\.")].copy()
    if not len(fn):
        return
    fn["k"] = fn["series_id"].str.split(".").str[3]
    geo["k"] = geo["series_id"].str.split(".").str[3]
    total = int(snap[snap["series_id"] == "s2.greenhouse.total_open_roles"]["value"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fn = fn.sort_values("value")
    ax1.barh(fn["k"], fn["value"], color="#632ca6")
    ax1.set_title(f"S2 · Open roles by function (total = {total}, as of {d['date'].max()})")
    ax1.set_xlabel("open roles")
    geo = geo.sort_values("value")
    ax2.barh(geo["k"], geo["value"], color="#e8a33d")
    ax2.set_title("S2 · Open roles by geography")
    ax2.set_xlabel("open roles")
    fig.tight_layout()
    fig.savefig(FIG / "03_hiring_breakdown.png")
    plt.close(fig)
    print("  -> 03_hiring_breakdown.png")


def fig4_dol() -> None:
    """S2 历史：DOL H-1B/LCA 季度招聘强度 vs DDOG 收入。"""
    p = PROCESSED_DIR / "s2_dol_lca_quarterly.csv"
    if not p.exists():
        return
    dol = pd.read_csv(p)
    tgt = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    dol = dol.merge(tgt[["quarter", "revenue", "revenue_yoy_pct"]], on="quarter", how="left")
    dol = dol.sort_values("quarter")
    # 可比的同比区间
    dol["cases_yoy"] = dol["lca_cases"].pct_change(4) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = dol["quarter"]
    ax1.bar(x, dol["lca_cases"], color="#632ca6", alpha=0.85, label="LCA cases (all)")
    ax1.bar(x, dol["lca_engineering_cases"], color="#a06cd5", alpha=0.9, label="engineering roles")
    ax1.set_title("S2 history · Datadog H-1B (LCA) filings per quarter")
    ax1.set_ylabel("cases")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=90)

    ok = dol.dropna(subset=["revenue_yoy_pct"])
    ax2.plot(ok["quarter"], ok["revenue_yoy_pct"], "s--", color="#e8a33d",
             label="DDOG revenue YoY %")
    ax2.plot(ok["quarter"], ok["cases_yoy"], "o-", color="#632ca6", label="LCA cases YoY %")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("YoY (%)")
    ax2.set_title("Hiring signal vs revenue growth")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(FIG / "04_dol_hiring_vs_revenue.png")
    plt.close(fig)
    print("  -> 04_dol_hiring_vs_revenue.png")


def fig5_headcount() -> None:
    """10-K 年度人力锚点。"""
    p = PROCESSED_DIR / "s2_headcount_annual.csv"
    if not p.exists():
        return
    hc = pd.read_csv(p).dropna(subset=["employees"]).sort_values("fiscal_year")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(hc["fiscal_year"], hc["employees"], color="#632ca6", alpha=0.85, label="total employees")
    if "employees_rnd" in hc.columns:
        ax.bar(hc["fiscal_year"], hc["employees_rnd"], color="#e8a33d", alpha=0.9,
               label="R&D employees")
    for _, r in hc.iterrows():
        ax.text(r["fiscal_year"], r["employees"], f"{r['employees']:,.0f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title("S2 history · Datadog headcount (10-K disclosures)")
    ax.set_ylabel("employees")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "05_headcount.png")
    plt.close(fig)
    print("  -> 05_headcount.png")


def main() -> None:
    print("生成图表 ->", FIG.relative_to(ROOT))
    for f in (fig1_npm, fig2_vs_revenue, fig3_hiring, fig4_dol, fig5_headcount):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"  ! {f.__name__} 失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
