"""统一分析：把 S1 + S2 + 宏观基本面 + 云厂商景气 合并成一张季度面板，并做严谨的信号评估。

背景
----
项目里有三路信号，此前由不同会话分别采集：
  S1 开发者插桩流量      npm SDK 下载量（日频，2016 起）
  S2 招聘 / 人力         Greenhouse 在招岗位（日频，前向）+ DOL H-1B（季频，2019Q4 起）+ 10-K 员工数（年频）
  宏观基本面             FRED/ALFRED 18 个序列（**ALFRED vintage 时点口径，无修订偏差**）
  云厂商景气             AMZN/MSFT/GOOGL 收入与资本开支（SEC XBRL，带 first_filed）

本脚本把它们合并到同一个面板上，用**同一套评估协议**衡量每一个信号，避免各说各话。

评估协议（本项目的方法学核心）
------------------------------
1. **时点对齐**：季度 t 的特征只使用「该季财报发布日前一天」已可得的数据
   （宏观用 ALFRED vintage，云厂商用 first_filed，npm 用部分季度同比窗口）。
2. **两个预测口径**：
   - 水平口径  y   = 收入同比增速
   - 加速口径  Δy  = 本季增速 − 上季增速（"加速/减速"的方向判断）
3. **朴素基准（必须打败的对手）**：水平口径的上季增速；加速口径的 Δy=0。
4. **方向命中率必须对"基准率"检验**：若窗口内 7/12 是加速，那么无脑猜"加速"就有 58.3%，
   任何低于/接近该值的方向命中率都没有信息量。用单边二项检验给出 p 值。
5. **配对 bootstrap**：对每个模型与朴素基准的误差差做重采样，给出置信区间与"优于基准"的概率。
6. **样本量与检出力**：OOS 仅 10–16 个季度，必须显式声明"未发现显著改善 ≠ 不存在关系"。

产出
----
  data/processed/unified_panel.csv             统一季度特征面板
  data/processed/unified_factor_screening.csv  每个候选因子的单因子走步回测
  data/processed/unified_eval_metrics.csv      模型指标（水平 + 加速两个口径）
  docs/model_findings.md                       结论报告

用法：.venv/bin/python scripts/unified_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, ROOT, write_table  # noqa: E402

RNG = np.random.default_rng(20260914)
MIN_TRAIN = 12          # 走步回测的最小训练窗（季度）
PARTIAL_FLAG = "npm_partial"

# 宏观因子的分组（来自 macro_catalog.csv，用于分层报告）
MACRO_BLOCKS = {
    "A 宏观需求": ["GDPC1", "A191RL1Q225SBEA", "INDPRO", "PAYEMS", "NEWORDER", "CFNAI"],
    "B 利率流动性": ["DGS10", "DFII10", "FEDFUNDS", "T10Y2Y"],
    "C 金融条件": ["NFCI"],
    "D 通胀汇率": ["CPIAUCSL", "DTWEXBGS"],
    "E 行业景气": ["IPG3344S"],
    "F 景气调查": ["GACDFSA066MSFRBPHI", "GACDISA066MSFRBNY"],
}
BLOCK_OF = {f"{s}_{t}": b for b, ss in MACRO_BLOCKS.items() for s in ss for t in ("lvl", "yoy")}


# --------------------------------------------------------------------------- #
# 1. 统一面板
# --------------------------------------------------------------------------- #
def npm_quarterly() -> pd.DataFrame:
    """从 S1 日频观测重建 npm 季度口径（独立于其他会话的实现，便于交叉验证）。

    关键：季度未结束时用「同天数窗口」的去年同期做分母，避免部分季度导致的同比失真。
    """
    obs = pd.read_parquet(PROCESSED_DIR / "observations.parquet")
    npm = obs[obs["series_id"].str.startswith("s1.npm.")].copy()
    npm["date"] = pd.to_datetime(npm["date"])
    daily = npm.groupby("date")["value"].sum().sort_index()          # 全部 npm 包合计

    rows = []
    for q, g in daily.groupby(daily.index.to_period("Q")):
        start, end = q.start_time, q.end_time
        observed = g.index
        partial = bool(observed.max() < end)
        cur = float(g.sum())
        # 基期：去年同期、同样的「月-日」窗口
        base_mask = ((daily.index >= start - pd.DateOffset(years=1)) &
                     (daily.index <= observed.max() - pd.DateOffset(years=1)))
        base = float(daily[base_mask].sum()) if base_mask.any() else np.nan
        rows.append({"quarter": str(q), "period_end": end.date(),
                     "npm_total": cur, "npm_days_observed": len(observed),
                     "npm_partial": partial,
                     "npm_yoy": (cur / base - 1) * 100 if base and base > 0 else np.nan})
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    tgt = pd.read_csv(PROCESSED_DIR / "ddog_target_quarterly.csv")
    tgt["period_end"] = pd.to_datetime(tgt["period_end"])
    tgt["quarter"] = tgt["quarter"].astype(str)

    # 目标变量
    y = tgt[["quarter", "period_end", "revenue", "revenue_yoy_pct", "billings", "rpo",
             "customers_100k", "filing_date"]].rename(columns={"revenue_yoy_pct": "y"})

    # S1
    npm = npm_quarterly()
    npm["period_end"] = pd.to_datetime(npm["period_end"])
    panel = y.merge(npm, on=["quarter", "period_end"], how="left")

    # 宏观 + 云厂商：使用已通过时点校验的面板（ALFRED vintage / first_filed）
    ref = pd.read_csv(PROCESSED_DIR / "forecast_panel.csv")
    ref["quarter"] = ref["quarter"].astype(str)
    macro_cols = [c for c in ref.columns if c in BLOCK_OF]
    keep = ["quarter", "vintage"] + macro_cols + \
           [c for c in ["cloud_rev_yoy", "cloud_capex_yoy"] if c in ref.columns]
    panel = panel.merge(ref[keep], on="quarter", how="left")

    # 云厂商绝对值（用于分位/背景，不直接入模）
    cq = pd.read_csv(PROCESSED_DIR / "cloud_complex_panel.csv")
    cq["quarter"] = cq["quarter"].astype(str)
    panel = panel.merge(cq[["quarter", "revenue_sum", "capex_sum"]], on="quarter", how="left")

    # S2 · DOL H-1B 季度
    dol = pd.read_csv(PROCESSED_DIR / "s2_dol_lca_quarterly.csv")
    dol["quarter"] = dol["quarter"].astype(str)
    dm = dol.rename(columns={c: f"dol_{c}" for c in dol.columns if c != "quarter"})
    panel = panel.merge(dm, on="quarter", how="left")

    # S2 · 10-K 员工数（年度，按财年向下填充到四个季度 —— 建模时只用"上一年已披露"的值）
    hc = pd.read_csv(PROCESSED_DIR / "s2_headcount_annual.csv")
    hc = hc.dropna(subset=["employees"])[["fiscal_year", "employees", "employees_rnd", "countries"]]
    hc["fiscal_year"] = hc["fiscal_year"].astype(int)
    panel["_fy"] = panel["period_end"].dt.year - 1          # 12/31 期末，归属上一财年
    panel = panel.merge(
        hc.rename(columns={"employees": "hc_employees", "employees_rnd": "hc_rnd",
                           "countries": "hc_countries"}),
        left_on="_fy", right_on="fiscal_year", how="left").drop(columns=["_fy", "fiscal_year"])

    # S2 · Greenhouse 日频（前向累积，仅作为仪表盘监测，不进入建模特征）
    gh = pd.read_csv(PROCESSED_DIR / "s2_hiring_daily.csv")
    gh = gh[gh["series_id"] == "s2.greenhouse.total_open_roles"]
    if len(gh):
        gh["quarter"] = pd.to_datetime(gh["date"]).dt.to_period("Q").astype(str)
        g = gh.groupby("quarter")["value"].mean().rename("gh_open_roles")
        panel = panel.merge(g, on="quarter", how="left")

    panel = panel.sort_values("quarter").reset_index(drop=True)
    # 目标的时间序列派生量
    panel["y_lag1"] = panel["y"].shift(1)
    panel["y_lag2"] = panel["y"].shift(2)
    panel["dy"] = panel["y"] - panel["y_lag1"]
    panel["npm_accel"] = panel["npm_yoy"] - panel["npm_yoy"].shift(1)
    panel["hc_yoy"] = panel["hc_employees"].pct_change(4) * 100
    return panel


# --------------------------------------------------------------------------- #
# 2. 走步回测
# --------------------------------------------------------------------------- #
def walk_forward(panel: pd.DataFrame, feats: list[str], target: str,
                 oos_from: str, ridge: bool = False) -> pd.DataFrame:
    d = panel.dropna(subset=[target]).reset_index(drop=True)
    idx = d.index[d["quarter"] >= oos_from]
    if not len(idx):
        return pd.DataFrame()
    out = []
    for i in idx:
        tr = d.iloc[:i].dropna(subset=feats + [target])
        if len(tr) < MIN_TRAIN:
            continue
        model = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if ridge
                 else LinearRegression())
        model.fit(tr[feats], tr[target])
        x = d.iloc[[i]][feats].copy()
        for c in feats:                                   # 缺失用训练集中位数兜底
            if pd.isna(x[c].iloc[0]):
                x[c] = tr[c].median()
        out.append({"quarter": d.loc[i, "quarter"], "y_true": d.loc[i, target],
                    "y_prev": d.loc[i, "y_lag1"], "y_pred": float(model.predict(x)[0])})
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame, name: str, target: str) -> dict:
    """误差指标 + **对基准率检验过的**方向命中率。"""
    if not len(df):
        return {}
    e = df["y_true"] - df["y_pred"]
    res = {"model": name, "target": target, "n_oos": len(df),
           "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()), "bias": e.mean()}
    if target == "y":
        res["mape_pct"] = (e.abs() / df["y_true"].abs()).mean() * 100
        # 方向判断 = 「相对上季是加速还是减速」，而不是「增速是否为正」
        # （DDOG 各季增速全为正，用 sign(y_true) 会得到恒为 +1 的无意义结果）
        act = np.sign(df["y_true"] - df["y_prev"])
        pred = np.sign(df["y_pred"] - df["y_prev"])
    else:
        res["mape_pct"] = np.nan
        act = np.sign(df["y_true"])
        pred = np.sign(df["y_pred"])
    calls = pred != 0
    res["n_dir_calls"] = int(calls.sum())
    res["dir_hit_pct"] = float((pred[calls] == act[calls]).mean() * 100) if calls.sum() else np.nan
    # 基准率：窗口内多数方向的占比（无脑猜多数方向的命中率）
    if len(df):
        res["base_rate_pct"] = float(max((act > 0).mean(), (act < 0).mean()) * 100)
        res["edge_vs_base_pp"] = (res["dir_hit_pct"] - res["base_rate_pct"]) if calls.sum() else np.nan
        res["dir_pvalue"] = _binom_p(int((pred[calls] == act[calls]).sum()), int(calls.sum()),
                                     res["base_rate_pct"] / 100) if calls.sum() else np.nan
    return res


def _binom_p(k: int, n: int, p0: float) -> float:
    """单边二项检验：P(X >= k)，原假设为基准率 p0。"""
    from math import comb
    if n == 0:
        return np.nan
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


def paired_bootstrap(model_df: pd.DataFrame, naive_df: pd.DataFrame,
                     target: str, n_boot: int = 4000) -> dict:
    """配对 bootstrap：模型误差 − 朴素基准误差（负=更好）。"""
    m = model_df.merge(naive_df, on="quarter", suffixes=("_m", "_n"))
    if len(m) < 6:
        return {}
    if target == "y":
        em = (m["y_true_m"] - m["y_pred_m"]).abs() / m["y_true_m"].abs()
        en = (m["y_true_n"] - m["y_pred_n"]).abs() / m["y_true_n"].abs()
    else:
        em = (m["y_true_m"] - m["y_pred_m"]) ** 2
        en = (m["y_true_n"] - m["y_pred_n"]) ** 2
    d = (em - en).to_numpy()
    boot = np.array([RNG.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return {"n_pairs": len(d), "d_mean": float(d.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "prob_better": float((boot < 0).mean())}


# --------------------------------------------------------------------------- #
# 3. 主流程
# --------------------------------------------------------------------------- #
def baseline_paired(r: pd.DataFrame, target: str) -> pd.DataFrame:
    """在**与模型完全相同的季度**上构造朴素基准。

    这一步是必须的：走步回测会因为 MIN_TRAIN 限制而丢掉窗口早期的若干季度，
    若把基准算在整个窗口上，就变成了"模型在容易的样本上、基准在困难的样本上"
    ——实测这会凭空造出 4pp 以上的假优势（曾把 6.51pp 的基准错配给 2.63pp 的样本）。
    """
    b = r[["quarter", "y_true", "y_prev"]].copy()
    b["y_pred"] = b["y_prev"] if target == "y" else 0.0
    return b


def main() -> None:
    panel = build_panel()
    print("== 统一季度面板 ==")
    print(f"  {panel.shape[0]} 个季度 × {panel.shape[1]} 列  "
          f"（{panel['quarter'].iloc[0]} → {panel['quarter'].iloc[-1]}）")
    for p in write_table(panel, "unified_panel"):
        print("  ->", p)

    # 候选因子清单
    s1 = [c for c in ["npm_yoy", "npm_accel"] if c in panel.columns]
    s2 = [c for c in ["dol_lca_cases", "dol_lca_new_employment", "dol_lca_engineering_cases",
                      "dol_lca_cases_yoy", "hc_yoy", "hc_employees"]
          if c in panel.columns]
    macro = [c for c in panel.columns if c in BLOCK_OF]
    cloud = [c for c in ["cloud_rev_yoy", "cloud_capex_yoy"] if c in panel.columns]
    blocks = {"S1 插桩流量": s1, "S2 招聘/人力": s2, "云厂商景气": cloud, "宏观基本面": macro}

    print("\n== 候选因子 ==")
    for b, cols in blocks.items():
        print(f"  {b}: {len(cols)} 个")

    oos_windows = ["2022Q3", "2023Q3"]
    screen_rows, metric_rows, boot_rows = [], [], []

    for oos_from in oos_windows:
        for target, tname in [("y", "水平(增速)"), ("dy", "加速(Δ增速)")]:
            feats_possible = s1 + s2 + macro + cloud
            # 单因子的正确测法：在「上季增速」这个 AR 锚之上衡量增量信息。
            # 若把因子单独丢进 OLS 会严重误设——实测 npm 单因子的预测值恒在 40%~55%，
            # 而实际增速只有 25%~35%，方向判断退化成「永远喊加速」，
            # 命中率必然等于基准率（58.33%），毫无信息量。
            ar_anchor = ["y_lag1"] if target == "y" else []
            # 朴素基准不再单独构造：每个模型都用 baseline_paired() 在**同一批季度**上配对比较
            if target == "dy":                     # 额外测一个「均值回复」基准
                r_ar = walk_forward(panel, ["y_lag1"], target, oos_from)
                if len(r_ar):
                    m_ar = metrics(r_ar, "⑧ 上季增速（均值回复）", target)
                    m_ar.update({"oos_from": oos_from, "target_label": tname, "block": "基准"})
                    b = paired_bootstrap(r_ar, baseline_paired(r_ar, target), target)
                    m_ar.update({f"boot_{k}": v for k, v in b.items()})
                    metric_rows.append(m_ar)

            for bname, cols in blocks.items():
                for c in cols:
                    r = walk_forward(panel, ar_anchor + [c], target, oos_from)
                    if not len(r):
                        continue
                    mm = metrics(r, f"{c}", target)
                    mm.update({"oos_from": oos_from, "target_label": tname, "block": bname})
                    b = paired_bootstrap(r, baseline_paired(r, target), target)
                    mm.update({f"boot_{k}": v for k, v in b.items()})
                    screen_rows.append(mm)

            for bname, cols in blocks.items():
                cols = [c for c in cols if c in feats_possible]
                if not cols:
                    continue
                r = walk_forward(panel, ar_anchor + cols, target, oos_from, ridge=True)
                if not len(r):
                    continue
                mm = metrics(r, f"[组合] {bname}", target)
                mm.update({"oos_from": oos_from, "target_label": tname, "block": bname})
                b = paired_bootstrap(r, baseline_paired(r, target), target)
                mm.update({f"boot_{k}": v for k, v in b.items()})
                screen_rows.append(mm)

            allcols = [c for c in feats_possible]
            r = walk_forward(panel, ar_anchor + allcols, target, oos_from, ridge=True)
            if len(r):
                mm = metrics(r, "[组合] 全部信号 ridge", target)
                mm.update({"oos_from": oos_from, "target_label": tname, "block": "全部"})
                b = paired_bootstrap(r, baseline_paired(r, target), target)
                mm.update({f"boot_{k}": v for k, v in b.items()})
                screen_rows.append(mm)

    scr = pd.DataFrame(screen_rows)
    met = pd.DataFrame(metric_rows)
    scr = pd.concat([scr, met], ignore_index=True)

    print("\n== 单因子筛选（按 RMSE 排序，水平口径 / OOS 自 2023Q3）==")
    show = scr[(scr["target"] == "y") & (scr["oos_from"] == "2023Q3")].copy()
    show = show.sort_values("rmse")
    cols = ["model", "block", "n_oos", "mape_pct", "rmse", "dir_hit_pct", "base_rate_pct",
            "edge_vs_base_pp", "dir_pvalue", "boot_prob_better"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].round(3).to_string(index=False))

    for p in write_table(scr, "unified_factor_screening"):
        print("  ->", p)

    # 结论摘要
    print("\n" + "=" * 96)
    print("严格配对比较：朴素基准在**每个模型自己的 OOS 季度集**上重算")
    print("=" * 96)
    for oos_from in oos_windows:
        for target, tname in [("y", "水平(增速)"), ("dy", "加速(Δ增速)")]:
            sub = scr[(scr["target"] == target) & (scr["oos_from"] == oos_from)].dropna(subset=["rmse"])
            if not len(sub):
                continue
            best = sub.sort_values("rmse").iloc[0]
            nb = scr[(scr["target"] == target) & (scr["oos_from"] == oos_from) &
                     (scr["model"].str.startswith("⑧"))]
            win = "✅ 击败" if best.get("boot_prob_better", 1) > 0.95 else "❌ 未能在统计上击败"
            print(f"OOS≥{oos_from} | {tname}: 最好信号「{best['model']}」"
                  f"RMSE={best['rmse']:.2f}pp  MAPE={best['mape_pct']:.2f}%  "
                  f"方向{best['dir_hit_pct']:.1f}%(基准率{best['base_rate_pct']:.1f}%)  "
                  f"P(优于配对朴素)={best.get('boot_prob_better', float('nan')):.3f}  → {win}")
    print("=" * 96)


if __name__ == "__main__":
    main()
