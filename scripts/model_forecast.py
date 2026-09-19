"""因子增强的时序预测：因子分析 → 简单模型 → 走步回测（项目的"第 2 步"）。

目标 y：DDOG 季度收入同比增速（%），并换算回收入水平做 MAPE。
三类特征
--------
  ① 公司私有高频信号：npm SDK 下载量（S1，日频 → 季度，2016 至今）
  ② **宏观环境因子**：美国联邦公开数据（GDP / 工业生产 / 核心资本开支订单 / 利率 /
     金融条件 / 通胀 / 美元），13 个序列、4 个大类
  ③ 行业景气因子：AMZN+MSFT+GOOGL 收入与资本开支合计（SEC XBRL）

方法论（本脚本的全部价值都在这里）
--------------------------------
1. **时点对齐，杜绝前视**：第 q 季度的特征只取"该季财报发布日前一天"真实可得的数据。
   - 宏观：ALFRED vintage（不是最新修订值）→ 消除 revision bias
   - 云厂商：XBRL 的 first_filed ≤ vintage
   - npm：只累加到 vintage 为止的日频数据；同比用**同窗口**（今年 vs 去年同样天数）避免机械偏差
2. **模型一律简单**（作业明确要求）：naive / AR(1) / OLS / 强正则 Ridge / PCA 宏观因子。
   样本只有 ~28 个季度，特征越多越容易过拟合，所以先做**单因子增量筛选**，再组合。
3. **走步回测**（扩展窗口）：预测每季只用它之前的信息训练；PCA 与标准化只在训练集上拟合。
4. **诚实汇报**：baseline 与增强模型并列，让"宏观因子到底有没有增量"由数字说话。

产出
----
  data/processed/forecast_panel.csv        季度特征矩阵（含每列来源）
  data/processed/forecast_metrics.csv      走步回测指标（模型 × 指标）
  data/processed/factor_screening.csv      单因子增量筛选排名 ← 回答"该纳入哪 3-4 个因子"
  data/processed/macro_pca_loadings.csv    宏观块 PCA 载荷（因子分析本体）
  data/processed/macro_pca_variance.csv    PCA 方差解释
  data/processed/leadlag.csv               互相关（领先/滞后阶数）
  data/processed/forecast_oos.csv          各模型逐季样本外预测
  data/processed/forecast_latest.csv       当前（未披露）季度实时 nowcast
  reports/figures/06_macro_factors.png 07_backtest.png 08_macro_pca.png

运行：
  export no_proxy="*" NO_PROXY="*"
  .venv/bin/python scripts/model_forecast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, ROOT, write_table  # noqa: E402
from collect_macro import CATALOG, ddog_vintages  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
# 中文字体：默认 DejaVu Sans 没有 CJK 字形，会把标题渲染成方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MIN_TRAIN = 12          # 初始训练窗口（季度）
RIDGE_ALPHA = 10.0      # 强正则：样本太少时不追求拟合优度
# npm 基期门槛：2018–2020 年 npm 上只有 dd-trace 一个新包在从≈0 起量
# （2018Q1 全库仅 177 次下载），这一段的同比是 30 万% 级别的假极值、完全不可比。
# 因此要求"去年同期窗口"至少有 100 万次下载（≈1.1 万次/日）才计算同比。
NPM_BASE_MIN = 1_000_000

# 每个宏观大类里"经济逻辑最直接"的代表因子 —— 最终 4 因子组合（可由筛选结果替换）
BLOCK_PICK = {
    "A 宏观需求": "NEWORDER_yoy",       # 核心资本开支订单同比：企业 IT 支出上游
    "B 利率流动性": "DFII10_lvl",        # 10Y 实际利率：真实贴现率
    "C 金融条件": "NFCI_lvl",            # 芝加哥联储金融条件指数：风险偏好/信用松紧
    "D 通胀汇率": "DTWEXBGS_yoy",        # 美元同比：海外收入的汇率逆风
}


# ---------------------------------------------------------------------------
def q_prev(q: str, k: int = 1) -> str:
    y, qq = int(q[:4]), int(q[-1])
    y2, qq2 = divmod(y * 4 + (qq - 1) - k, 4)
    return f"{y2}Q{qq2 + 1}"


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_target() -> pd.DataFrame:
    t = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    vmap = {q: str(v) for q, v in ddog_vintages()}
    t["vintage"] = t["quarter"].map(vmap)
    return t[["quarter", "period_end", "revenue", "revenue_yoy_pct", "customers_100k",
              "filing_date", "vintage"]].copy()


def load_macro_wide() -> pd.DataFrame:
    """时点对齐的宏观面板 → 宽表（每序列两个变换：季度均值水平 _lvl、同比 _yoy）。"""
    p = PROCESSED_DIR / "macro_panel_asof.csv"
    if not p.exists():
        raise SystemExit("缺少 macro_panel_asof.csv，请先运行 collect_macro.py")
    m = pd.read_csv(p)
    m["short"] = m["series_key"].str.replace("macro.fred.", "", regex=False)
    m = m.sort_values("vintage").drop_duplicates(["quarter", "short"], keep="last")
    lvl = m.pivot_table(index="quarter", columns="short", values="value", aggfunc="last")
    yoy = m.pivot_table(index="quarter", columns="short", values="value_yoy_pct", aggfunc="last")
    lvl.columns = [f"{c}_lvl" for c in lvl.columns]
    yoy.columns = [f"{c}_yoy" for c in yoy.columns]
    out = pd.concat([lvl, yoy], axis=1).reset_index()
    vint = m.groupby("quarter")["vintage"].max().rename("vintage_macro").reset_index()
    return out.merge(vint, on="quarter", how="left")


def _quarter_end(q: str) -> pd.Timestamp:
    y, qq = int(q[:4]), int(q[-1])
    return pd.Timestamp(y, qq * 3, 1) + pd.offsets.MonthEnd(0)


def load_npm(cframe: pd.DataFrame) -> pd.DataFrame:
    """npm 季度下载量：窗口 = 季度起点 → min(季末, vintage)，同比用**同窗口**去年值。"""
    obs = pd.read_parquet(PROCESSED_DIR / "observations.parquet")
    npm = obs[obs["series_id"].str.startswith("s1.npm.")].copy()
    npm["date"] = pd.to_datetime(npm["date"])
    npm["pkg"] = (npm["series_id"].str.replace("s1.npm.", "", regex=False)
                  .str.replace(".downloads", "", regex=False))

    rows = []
    for _, r in cframe.iterrows():
        q = r["quarter"]
        vint = pd.Timestamp(r["vintage"]) if pd.notna(r.get("vintage")) else pd.Timestamp("2100-01-01")
        start = _quarter_end(q_prev(q, 1)) + pd.Timedelta(days=1)
        end_q = _quarter_end(q)
        end = min(end_q, vint)                       # 只用到 vintage 为止
        win = npm[(npm["date"] >= start) & (npm["date"] <= end)]
        win_prev = npm[(npm["date"] >= start - pd.DateOffset(years=1))
                       & (npm["date"] <= end - pd.DateOffset(years=1))]
        if not len(win):
            continue
        tot, tot_prev = win["value"].sum(), win_prev["value"].sum()
        dt = win.groupby("pkg")["value"].sum()
        dtp = win_prev.groupby("pkg")["value"].sum()
        ok_base = tot_prev >= NPM_BASE_MIN                    # 基期太小 → 同比不可比
        rows.append({"quarter": q, "npm_total": float(tot),
                     "npm_ddtrace": float(dt.get("dd-trace", np.nan)),
                     "npm_base_ok": bool(ok_base),
                     "npm_yoy": ((tot / tot_prev - 1) * 100) if (tot_prev and ok_base) else np.nan,
                     "npm_ddtrace_yoy": ((dt.get("dd-trace", np.nan) / dtp.get("dd-trace", np.nan) - 1) * 100
                                         if (dtp.get("dd-trace", 0) and ok_base) else np.nan),
                     "npm_days_observed": int(win["date"].nunique()),
                     "npm_partial": bool(end < end_q)})
    d = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    # q/q 环比（对数差分）：不依赖"去年同期"，因此不受基期过小影响，样本可往前延伸。
    # 仅在两期都完整时才计算（当前未结束的季度留空）。
    prev_tot = d["npm_total"].shift(1)
    both_full = (~d["npm_partial"]) & (~d["npm_partial"].shift(1).fillna(True))
    d["npm_qq"] = np.where(both_full & (prev_tot > 0), np.log(d["npm_total"] / prev_tot) * 100, np.nan)
    return d


def load_cloud_asof(cframe: pd.DataFrame) -> pd.DataFrame:
    """云厂商景气因子：只取 first_filed ≤ 该季 vintage 的季度值（防前视）。"""
    p = PROCESSED_DIR / "cloud_complex_panel.csv"
    if not p.exists():
        print("  ! 缺 cloud_complex_panel.csv，跳过行业景气因子")
        return pd.DataFrame(columns=["quarter"])
    c = pd.read_csv(p)
    c["first_filed"] = pd.to_datetime(c["first_filed"], errors="coerce")
    keep = []
    for _, r in cframe.iterrows():
        vint = pd.Timestamp(r["vintage"]) if pd.notna(r.get("vintage")) else pd.Timestamp("2100-01-01")
        hit = c[(c["quarter"] == r["quarter"]) & (c["first_filed"] <= vint)]
        if len(hit):
            h = hit.iloc[[0]].copy()
            keep.append({"quarter": r["quarter"],
                         "cloud_rev_yoy": float(h["revenue_yoy_pct"].iloc[0]),
                         "cloud_capex_yoy": float(h["capex_yoy_pct"].iloc[0])})
    return pd.DataFrame(keep) if keep else pd.DataFrame(columns=["quarter"])


def build_panel() -> pd.DataFrame:
    """季度特征矩阵：已披露季度 + 当前未披露季度（用于实时 nowcast）。"""
    t = load_target()
    macro = load_macro_wide()

    # 未披露季度（宏观 live vintage 里有、目标表里没有）
    extra = macro[~macro["quarter"].isin(t["quarter"])].copy()
    if len(extra):
        ex = extra[["quarter", "vintage_macro"]].rename(columns={"vintage_macro": "vintage"})
        ex["period_end"] = ex["quarter"].map(lambda q: _quarter_end(q).date().isoformat())
        ex["revenue"] = np.nan
        ex["revenue_yoy_pct"] = np.nan
        ex["customers_100k"] = np.nan
        ex["filing_date"] = pd.NaT
        t = pd.concat([t, ex[t.columns]], ignore_index=True)

    cframe = t.sort_values("quarter").reset_index(drop=True)
    p = cframe.merge(load_npm(cframe), on="quarter", how="left")
    p = p.merge(macro.drop(columns=["vintage_macro"]), on="quarter", how="left")
    cloud = load_cloud_asof(cframe)
    if len(cloud):
        p = p.merge(cloud, on="quarter", how="left")

    p = p.sort_values("quarter").reset_index(drop=True)
    p["y"] = p["revenue_yoy_pct"]
    for k in (1, 2, 4):
        p[f"y_lag{k}"] = p["y"].shift(k)
    p["revenue_lag4"] = p["revenue"].shift(4)
    return p


# ---------------------------------------------------------------------------
# 走步回测
# ---------------------------------------------------------------------------
def _make_model(kind: str):
    return Ridge(alpha=RIDGE_ALPHA) if kind == "ridge" else LinearRegression()


def walk_forward(panel: pd.DataFrame, feats: list[str], kind: str = "ols",
                 pca_cols: list[str] | None = None, n_pc: int = 1,
                 min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """扩展窗口走步回测：预测第 i 季只用第 i 季之前的数据训练；PCA/标准化只在训练集拟合。"""
    cols = list(dict.fromkeys(feats + [c for c in (pca_cols or [])]))
    # 只保留"特征与标签都完整"的季度：这样 2019Q3（首个有同比、但无上季同比）等
    # 不完整行会被自动剔除，而不是让所有训练折全部作废。
    d = panel.dropna(subset=["y"] + cols).reset_index(drop=True)
    recs = []
    for i in range(min_train, len(d)):
        train, test = d.iloc[:i], d.iloc[[i]]
        if train[cols].isna().any().any() or test[cols].isna().any().any():
            continue
        tr, te = train.copy(), test.copy()
        use = list(feats)
        if pca_cols:
            sc = StandardScaler().fit(tr[pca_cols].values)
            pc = PCA(n_components=n_pc).fit(sc.transform(tr[pca_cols].values))
            for j in range(n_pc):
                tr[f"PC{j + 1}"] = pc.transform(sc.transform(tr[pca_cols].values))[:, j]
                te[f"PC{j + 1}"] = pc.transform(sc.transform(te[pca_cols].values))[:, j]
            use += [f"PC{j + 1}" for j in range(n_pc)]
        model = _make_model(kind).fit(tr[use].values, tr["y"].values)
        recs.append({"quarter": test["quarter"].iloc[0],
                     "y_true": float(test["y"].iloc[0]),
                     "y_pred": float(model.predict(te[use].values)[0]),
                     "y_prev": float(train["y"].iloc[-1])})
    return pd.DataFrame(recs)


def naive_walk(panel: pd.DataFrame, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    d = panel.dropna(subset=["y"]).reset_index(drop=True)
    recs = []
    for i in range(min_train, len(d)):
        prev = float(d["y"].iloc[i - 1])
        recs.append({"quarter": d["quarter"].iloc[i], "y_true": float(d["y"].iloc[i]),
                     "y_pred": prev, "y_prev": prev})
    return pd.DataFrame(recs)


def metrics(r: pd.DataFrame) -> dict:
    if not len(r):
        return {}
    e = r["y_pred"] - r["y_true"]
    call = (r["y_pred"] - r["y_prev"]).abs() > 1e-9      # 做出了方向判断的季度
    hit = ((np.sign(r["y_pred"] - r["y_prev"]) == np.sign(r["y_true"] - r["y_prev"])) & call)
    return {"n_oos": len(r),
            "mape_pct": float(np.mean(np.abs(e / r["y_true"])) * 100),
            "rmse_pp": float(np.sqrt(np.mean(e ** 2))),
            "mae_pp": float(np.mean(np.abs(e))),
            "bias_pp": float(np.mean(e)),
            "n_dir_calls": int(call.sum()),
            "dir_hit_pct": float(hit.sum() / call.sum() * 100) if call.sum() else float("nan"),
            "mape_level_pct": float(np.mean(np.abs(((100 + r["y_pred"]) - (100 + r["y_true"]))
                                                    / (100 + r["y_true"]))) * 100)}


def screen_macro(panel: pd.DataFrame, macro_cols: list[str], base_feats: list[str]) -> pd.DataFrame:
    """逐个宏观因子做增量测试：baseline(AR1+S1) vs baseline+该因子。"""
    bm = metrics(walk_forward(panel, base_feats))
    rows = []
    for c in macro_cols:
        m = metrics(walk_forward(panel, base_feats + [c]))
        if not m:
            continue
        base_id = c.rsplit("_", 1)[0]
        blk = next((e["block"] for e in CATALOG if e["id"] == base_id), "-")
        rows.append({"factor": c, "block": blk, **m,
                     "d_mape_pct": m["mape_pct"] - bm["mape_pct"],
                     "d_rmse_pp": m["rmse_pp"] - bm["rmse_pp"],
                     "d_dir_hit_pct": m["dir_hit_pct"] - bm["dir_hit_pct"]})
    out = pd.DataFrame(rows).sort_values("d_mape_pct")
    out.attrs["baseline"] = bm
    return out


def leadlag(panel: pd.DataFrame, cols: list[str], max_lag: int = 4) -> pd.DataFrame:
    """互相关（k>0 表示指标领先收入 k 个季度：用指标 t-k 解释收入 t）。

    同时给出两种口径，因为**趋势序列的水平值互相关极易产生伪相关**：
      - level：同比水平值之间的相关（易被共同趋势抬高，只能当参考）
      - diff ：一阶差分（Δ同比）之间的相关 —— 剔趋势后才是"领先"的较可信证据
    """
    rows = []
    for c in cols:
        for k in range(-max_lag, max_lag + 1):
            m = pd.concat([panel[c], panel["y"].shift(k)], axis=1).dropna()
            if len(m) >= 8:
                rows.append({"series": c, "lag_quarters": k, "transform": "level",
                             "corr": float(m.iloc[:, 0].corr(m.iloc[:, 1])), "n": len(m)})
            dm = pd.concat([panel[c].diff(), panel["y"].shift(k).diff()], axis=1).dropna()
            if len(dm) >= 8:
                rows.append({"series": c, "lag_quarters": k, "transform": "diff",
                             "corr": float(dm.iloc[:, 0].corr(dm.iloc[:, 1])), "n": len(dm)})
    return pd.DataFrame(rows)


def paired_bootstrap(oos: pd.DataFrame, base_model: str, n_boot: int = 4000,
                     seed: int = 7) -> pd.DataFrame:
    """相对基线的**配对自举**：给出 ΔMAPE 的置信区间与"优于基线"的频率。

    n≈12 的样本外季度太少，单点 MAPE 差异几乎无意义；自举区间与频率是更诚实的表达。
    """
    if not len(oos):
        return pd.DataFrame()
    abs_err = oos.assign(ape=lambda d: (d["y_pred"] - d["y_true"]).abs() / d["y_true"].abs() * 100)
    base = abs_err[abs_err["model"] == base_model].set_index("quarter")["ape"]
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in abs_err.groupby("model"):
        s = g.set_index("quarter")["ape"]
        common = s.index.intersection(base.index)
        if len(common) < 6:
            continue
        d = (s.loc[common] - base.loc[common]).values
        boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        rows.append({"model": name, "baseline": base_model, "n_pairs": len(common),
                     "d_mape_pp": float(d.mean()),
                     "boot_ci_lo": float(np.percentile(boot, 5)),
                     "boot_ci_hi": float(np.percentile(boot, 95)),
                     "prob_better_than_baseline": float((boot < 0).mean())})
    return pd.DataFrame(rows).sort_values("d_mape_pp")


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def fig_macro(panel: pd.DataFrame) -> None:
    d = panel.dropna(subset=["y"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0][0]
    ax.plot(d["quarter"], d["y"], "s--", color="#e8a33d", label="DDOG 收入同比 %")
    ax.plot(d["quarter"], d["npm_yoy"], "o-", color="#632ca6", label="npm 下载量同比 %")
    ax.set_title("① 公司私有信号 vs 收入增速")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90)

    ax = axes[0][1]
    for c, col in [("NEWORDER_yoy", "#1f77b4"), ("DFII10_lvl", "#d62728")]:
        if c in d:
            ax.plot(d["quarter"], d[c], "o-", color=col, label=c, lw=1.3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("② 宏观需求（资本开支订单）与真实利率")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90)

    ax = axes[1][0]
    for c in ["NFCI_lvl", "DTWEXBGS_yoy", "A191RL1Q225SBEA_lvl"]:
        if c in d:
            ax.plot(d["quarter"], d[c], "o-", label=c, lw=1.3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("③ 金融条件 / 美元同比 / 美国 GDP 增速")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90)

    ax = axes[1][1]
    for c in ["cloud_rev_yoy", "cloud_capex_yoy"]:
        if c in d:
            ax.plot(d["quarter"], d[c], "o-", label=c, lw=1.3)
    ax.set_title("④ 行业景气：AMZN+MSFT+GOOGL 收入/资本开支同比")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90)

    fig.suptitle("宏观环境因子与公司信号（全部为时点口径，无前视）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "06_macro_factors.png")
    plt.close(fig)
    print("  -> 06_macro_factors.png")


def fig_backtest(preds: dict[str, pd.DataFrame], panel: pd.DataFrame, md: pd.DataFrame) -> None:
    if not len(md):
        print("  (跳过回测图：暂无样本外结果)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    base = panel.dropna(subset=["y"])
    ax1.plot(base["quarter"], base["y"], "k--", lw=1.6, label="实际收入同比 %")
    colors = ["#9aa0a6", "#1f77b4", "#632ca6", "#2ca02c", "#d62728"]
    for (name, r), c in zip(preds.items(), colors):
        if len(r):
            ax1.plot(r["quarter"], r["y_pred"], "o-", color=c, lw=1.4, label=name, ms=4)
    ax1.set_title("走步回测：收入同比增速预测 vs 实际")
    ax1.legend(fontsize=7.5)
    ax1.tick_params(axis="x", rotation=90)

    m = md.head(9)
    ax2.barh(m["model"], m["mape_pct"], color="#632ca6", alpha=0.85)
    for i, v in enumerate(m["mape_pct"]):
        ax2.text(v, i, f" {v:.2f}%", va="center", fontsize=8)
    ax2.set_xlabel("样本外 MAPE (%)，越低越好")
    ax2.set_title("各模型样本外误差")
    ax2.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "07_backtest.png")
    plt.close(fig)
    print("  -> 07_backtest.png")


def fig_pca(load: pd.DataFrame, evr: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    l = load.iloc[::-1]
    ax1.barh(l.index, l["PC1"], color="#632ca6", alpha=0.9)
    ax1.axvline(0, color="grey", lw=0.8)
    ax1.set_title(f"宏观因子 PC1 载荷（解释 {evr[0] * 100:.0f}% 方差）")
    ax1.tick_params(axis="y", labelsize=7)
    ax2.bar(range(1, len(evr) + 1), evr * 100, color="#e8a33d", label="单个")
    ax2.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, "ko-", ms=3, label="累计")
    ax2.set_xlabel("主成分")
    ax2.set_ylabel("解释方差 (%)")
    ax2.set_title("宏观块 PCA 碎石图")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "08_macro_pca.png")
    plt.close(fig)
    print("  -> 08_macro_pca.png")


# ---------------------------------------------------------------------------
def main() -> None:
    print("== 因子增强时序预测 ==")
    panel = build_panel()
    panel_full = panel.copy()                     # 含尚未披露的当前季度（nowcast 用）
    macro_cols = [c for c in panel.columns
                  if c.endswith(("_lvl", "_yoy")) and not c.startswith(("npm_", "cloud_"))]
    # 只保留在"有 y 的季度"里完全没有缺失的宏观因子（否则 PCA/OLS 会整段跳折）
    yidx = panel.index[panel["y"].notna()]
    dropped = [c for c in macro_cols if panel.loc[yidx, c].isna().any()]
    if dropped:
        print(f"   ! 剔除有缺失的宏观因子（不进模型）：{dropped}")
    macro_cols = [c for c in macro_cols if c not in dropped]
    n_y = int(panel["y"].notna().sum())
    print(f"   面板 {len(panel)} 行（其中已披露 {n_y} 个季度）；宏观因子候选 {len(macro_cols)} 个")
    base_feats = ["y_lag1", "npm_yoy"]
    # 统一样本：所有模型在同一批季度上评估（否则误差不可比）
    panel = panel.dropna(subset=["y"] + base_feats).reset_index(drop=True)
    print(f"   建模样本（同比可用 + 基期有意义）：{len(panel)} 个季度 "
          f"{panel['quarter'].iloc[0]} → {panel['quarter'].iloc[-1]}"
          f"（样本外起点 {panel['quarter'].iloc[MIN_TRAIN] if len(panel) > MIN_TRAIN else '-'}）")
    four = [c for c in BLOCK_PICK.values() if c in panel.columns]
    print(f"   4 因子组合（每大类一个代表）：{four}")

    # ---- 1) 单因子增量筛选 ----
    print("\n-- 单因子增量筛选（baseline = AR(1) + npm 同比）--")
    scr = screen_macro(panel, macro_cols, base_feats)
    bm = scr.attrs["baseline"]
    print(f"   baseline: MAPE {bm['mape_pct']:.2f}% | RMSE {bm['rmse_pp']:.2f}pp | "
          f"方向 {bm['dir_hit_pct']:.0f}% (n={bm['n_oos']})")
    print(scr[["factor", "block", "mape_pct", "d_mape_pct", "rmse_pp", "dir_hit_pct"]]
          .head(10).to_string(index=False))
    for p in write_table(scr, "factor_screening"):
        print("  ->", p)

    # ---- 2) 宏观块 PCA（因子分析本体；只在训练样本上拟合）----
    print("\n-- 宏观块 PCA（因子分析）--")
    d = panel.dropna(subset=["y"] + macro_cols).reset_index(drop=True)
    train_pca = d.iloc[:max(MIN_TRAIN, len(d) - 4)]
    sc = StandardScaler().fit(train_pca[macro_cols].values)
    pca = PCA(n_components=4).fit(sc.transform(train_pca[macro_cols].values))
    evr = pca.explained_variance_ratio_
    load = pd.DataFrame(pca.components_.T, index=macro_cols, columns=["PC1", "PC2", "PC3", "PC4"])
    print(f"   方差解释：PC1 {evr[0] * 100:.0f}% / PC2 {evr[1] * 100:.0f}% / "
          f"PC3 {evr[2] * 100:.0f}% / PC4 {evr[3] * 100:.0f}%")
    print("   PC1 载荷最高的 5 个（绝对值）：")
    print(load["PC1"].abs().sort_values(ascending=False).head(5).to_string())
    write_table(load.reset_index(names="factor"), "macro_pca_loadings")
    write_table(pd.DataFrame({"pc": [f"PC{i + 1}" for i in range(4)],
                              "explained_var_pct": evr * 100}), "macro_pca_variance")

    # ---- 3) 模型对比（走步回测）----
    print("\n-- 走步回测（扩展窗口，每季只用历史信息训练）--")
    preds: dict[str, pd.DataFrame] = {}
    preds["① naive(上季增速)"] = naive_walk(panel)
    preds["② AR(1)"] = walk_forward(panel, ["y_lag1"])
    preds["③ AR(1)+npm"] = walk_forward(panel, base_feats)
    preds["④ +美国GDP"] = walk_forward(panel, base_feats + ["GDPC1_yoy"])
    preds["⑤ +4宏观因子"] = walk_forward(panel, base_feats + four)
    preds["⑥ +4宏观(ridge)"] = walk_forward(panel, base_feats + four, kind="ridge")
    preds["⑦ +宏观PC1"] = walk_forward(panel, base_feats, pca_cols=macro_cols, n_pc=1)
    preds["⑧ +宏观PC1+PC2"] = walk_forward(panel, base_feats, pca_cols=macro_cols, n_pc=2)
    if "cloud_capex_yoy" in panel.columns:
        preds["⑨ +云厂商capex"] = walk_forward(panel, base_feats + ["cloud_capex_yoy"])
    all_feats = [c for c in ["y_lag1", "y_lag2", "npm_yoy", "npm_ddtrace_yoy"] + macro_cols
                 if c in panel.columns]
    preds["⑩ 全特征ridge"] = walk_forward(panel, all_feats, kind="ridge")

    rows, oos = [], []
    for name, r in preds.items():
        m = metrics(r)
        if m:
            rows.append({"model": name, **m})
        if len(r):
            oos.append(r.assign(model=name))
    md = pd.DataFrame(rows).sort_values("mape_pct")
    print(md.to_string(index=False))
    write_table(md, "forecast_metrics")
    oos_all = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    if len(oos_all):
        write_table(oos_all, "forecast_oos")

    # ---- 3b) 相对基线的配对自举（n≈12 下更诚实的表达）----
    base_name = "① naive(上季增速)"
    boot = paired_bootstrap(oos_all, base_name)
    if len(boot):
        print(f"\n-- 相对基线「{base_name}」的配对自举（ΔMAPE<0 = 优于基线）--")
        print(boot.round(3).to_string(index=False))
        write_table(boot, "forecast_delta_vs_naive")

    # ---- 3c) 稳健性：换用 q/q 环比信号（无基期门槛，样本更长）----
    print("\n-- 稳健性检验：q/q 环比信号（样本更长，2019Q4 起）--")
    qq_feats = ["y_lag1", "npm_qq"]
    panel_qq = panel_full.dropna(subset=["y"] + qq_feats).reset_index(drop=True)
    four_qq = [c for c in four if c in panel_qq.columns]
    print(f"   样本 {len(panel_qq)} 个季度 {panel_qq['quarter'].iloc[0]} → {panel_qq['quarter'].iloc[-1]}")
    preds_qq = {
        "① naive": naive_walk(panel_qq),
        "② AR(1)": walk_forward(panel_qq, ["y_lag1"]),
        "③ AR(1)+npm(q/q)": walk_forward(panel_qq, qq_feats),
        "④ +4宏观因子": walk_forward(panel_qq, qq_feats + four_qq),
        "⑤ +宏观PC1": walk_forward(panel_qq, qq_feats, pca_cols=macro_cols, n_pc=1),
    }
    rq, oq = [], []
    for name, r in preds_qq.items():
        m = metrics(r)
        if m:
            rq.append({"model": name, **m})
        if len(r):
            oq.append(r.assign(model=name))
    md_qq = pd.DataFrame(rq).sort_values("mape_pct")
    print(md_qq.to_string(index=False))
    write_table(md_qq, "forecast_metrics_qq")
    if oq:
        oq_all = pd.concat(oq, ignore_index=True)
        write_table(oq_all, "forecast_oos_qq")
        bq = paired_bootstrap(oq_all, "① naive")
        if len(bq):
            print(bq.round(3).to_string(index=False))
            write_table(bq, "forecast_delta_vs_naive_qq")

    # ---- 4) 领先滞后 ----
    ll = leadlag(panel, [c for c in ["npm_yoy", "NEWORDER_yoy", "GDPC1_yoy", "DFII10_lvl",
                                     "NFCI_lvl", "cloud_capex_yoy"] if c in panel.columns])
    write_table(ll, "leadlag")
    if len(ll):
        print("\n-- npm 信号的领先/滞后剖面（level = 同比水平值；diff = 一阶差分）--")
        print(ll[ll["series"] == "npm_yoy"][["lag_quarters", "transform", "corr", "n"]]
              .round(3).to_string(index=False))
        best = (ll[ll["transform"] == "diff"]
                .assign(a=lambda x: x["corr"].abs()).sort_values("a", ascending=False)
                .groupby("series").head(1))
        print("\n-- 差分口径下各指标最强相关（lag>0 = 指标领先收入）--")
        print(best[["series", "lag_quarters", "corr", "n"]].round(3).to_string(index=False))

    write_table(panel_full, "forecast_panel")

    # ---- 5) 当前季度实时 nowcast ----
    print("\n-- 当前季度 nowcast --")
    last = panel_full[panel_full["revenue"].isna()].tail(1)
    if not len(last):
        print("   目标表已覆盖到最新季度，无待预测季度")
    else:
        q = last["quarter"].iloc[0]
        # 候选模型：naive 不需要任何特征（因此永远可用），另两个需要信号/宏观因子
        cand = [("① naive(上季增速)", []), ("③ AR(1)+npm", base_feats),
                ("⑤ +4宏观因子", base_feats + four)]
        avail = [(nm, f) for nm, f in cand if not (f and last[f].isna().any().any())]
        if not avail or not len(md):
            print("   ! 当前季度特征缺失或样本外结果不足；跳过 nowcast")
        else:
            avail.sort(key=lambda t: float(md.loc[md["model"] == t[0], "mape_pct"].iloc[0]))
            model_name = avail[0][0]
            print(f"   {q} 可用特征集：{[n for n, _ in avail]} → 采用 OOS 误差最小的 {model_name}")
            base_rev = float(panel.loc[panel["quarter"] == q_prev(q, 4), "revenue"].iloc[0])
            rows_out = []
            for nm, f in avail:
                dtr = panel.dropna(subset=["y"] + f)
                rmse = float(md.loc[md["model"] == nm, "rmse_pp"].iloc[0])
                if not f:                                # naive：直接用上季实际增速
                    pred, spec = float(dtr["y"].iloc[-1]), "naive 基线（无需拟合）"
                else:
                    test = last.dropna(subset=f)
                    fit = _make_model("ols").fit(dtr[f].values, dtr["y"].values)
                    pred = float(fit.predict(test[f].values)[0])
                    spec = "OLS: " + " + ".join(f)
                rows_out.append({
                    "quarter": q, "as_of": str(last["vintage"].iloc[0]), "model": nm,
                    "selected": nm == model_name, "spec": spec,
                    "forecast_yoy_pct": round(pred, 2),
                    "band_pm_pp_1rmse": round(rmse, 2),
                    "forecast_yoy_low": round(pred - rmse, 2),
                    "forecast_yoy_high": round(pred + rmse, 2),
                    "last_reported_yoy_pct": float(dtr["y"].iloc[-1]),
                    "implied_revenue_usd": round(base_rev * (1 + pred / 100)),
                    "revenue_same_quarter_last_year": base_rev,
                    "npm_days_observed": int(last["npm_days_observed"].iloc[0]),
                    "npm_partial_quarter": bool(last["npm_partial"].iloc[0]),
                    "note": "季度未结束：npm 为季度至今口径（同比用同天数窗口），宏观为实时 vintage"})
            out = pd.DataFrame([r for r in rows_out if r["selected"]]
                               + [r for r in rows_out if not r["selected"]])
            print(out.T.to_string())
            write_table(out, "forecast_latest")

    # ---- 6) 图 ----
    print("\n-- 图表 --")
    fig_macro(panel)
    fig_backtest({k: preds[k] for k in ["② AR(1)", "③ AR(1)+npm", "⑤ +4宏观因子", "⑧ +宏观PC1+PC2"]
                  if k in preds}, panel, md)
    fig_pca(load, evr)
    print("\n完成。")


if __name__ == "__main__":
    main()
