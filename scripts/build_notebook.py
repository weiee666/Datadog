"""生成 notebooks/01_data_overview.ipynb，并预执行一遍把图表嵌进输出。

为什么不直接用 nbconvert：环境里没有 jupyter（用户自己有），但不影响我们
生成一个**已经带好输出**的 notebook —— 用一个小执行器逐 cell exec、
把 matplotlib 图形捕获成 base64 PNG 写回 ipynb，用户打开即可看到图。

用法：.venv/bin/python scripts/build_notebook.py
产出：
  notebooks/01_data_overview.ipynb    可直接用你自己的 Jupyter 打开 / 重跑
  reports/figures/notebook/*.png      同样的图，单独存了一份方便直接看
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "01_data_overview.ipynb"

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [

(MD, """# DDOG 高频 / 另类数据 · 数据总览

这个 notebook 用**最基本的图**把项目里采集到的全部数据摊开，回答一个问题：
**这些数据到底长什么样？**

**数据来源**（全部免费、公开、合法，详见 `docs/data_source_strategy.md`）：

| 区块 | 内容 | 频率 | 来源 |
|---|---|---|---|
| 目标变量 | DDOG 已披露的季度收入 / billings / RPO / $100k 客户数 | 季 | SEC EDGAR |
| S1 插桩流量 | npm / PyPI SDK 下载量（用量原料） | 日 | npm registry、pypistats |
| S2 招聘·人力 | 在招岗位 / H-1B 申请 / 10-K 员工数 | 日·季·年 | Greenhouse、美国劳工部、SEC |
| 宏观基本面 | 16 个美国宏观序列（ALFRED **时点口径**） | 日·周·月·季 | FRED / ALFRED |
| 云厂商景气 | AMZN+MSFT+GOOGL 收入与资本开支 | 季 | SEC XBRL |

**怎么运行**：只需 `pandas / numpy / matplotlib`，不需要联网、不需要 pyarrow。
如果你的 Jupyter 内核缺这几个包，把内核切到本项目自带的 `.venv` 即可。
"""),

(CODE, """import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 预执行时用的是无界面后端，plt.show() 会报一句无害的警告，这里静音
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------- 中文字体：macOS 自带的 PingFang / Heiti，找不到就自动回退（不报错） ----------
_available = {f.name for f in font_manager.fontManager.ttflist}
for _c in ["PingFang SC", "PingFang HK", "Heiti TC", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]:
    if _c in _available:
        matplotlib.rcParams["font.sans-serif"] = [_c] + list(matplotlib.rcParams["font.sans-serif"])
        print("使用中文字体:", _c)
        break
else:
    print("⚠️ 未找到中文字体，图上中文可能显示为方框")
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------- 定位项目根目录（无论从哪个目录启动 notebook 都能找到数据） ----------
def find_root() -> Path:
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "data" / "processed").exists():
            return cand
    raise FileNotFoundError("找不到项目根目录（需要包含 data/processed）")

ROOT = find_root()
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "reports" / "figures" / "notebook"
FIGDIR.mkdir(parents=True, exist_ok=True)
print("项目根目录:", ROOT)


def load(name: str, **kw) -> pd.DataFrame:
    \"\"\"读 data/processed/<name>.csv（用 utf-8-sig，避免 BOM 污染首列名）。\"\"\"
    return pd.read_csv(PROC / f"{name}.csv", encoding="utf-8-sig", **kw)


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight")
"""),

(MD, """---
## ① 目标变量：DDOG 已披露的季度 KPI

这是我们要预测的 y。注意两点：
1. 收入**每个季度都在同比增长**，增速在 25%–36% 之间移动；
2. 2026 年增速**重新加速**（29% → 32% → 36%），这是整个项目最关键的现实现象。
"""),

(CODE, """tgt = load("ddog_target_quarterly")
tgt["period_end"] = pd.to_datetime(tgt["period_end"])
t = tgt.dropna(subset=["revenue"]).tail(18)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(t["quarter"], t["revenue"] / 1e6, color="#632ca6", alpha=0.85, label="季度收入")
ax.set_ylabel("季度收入（百万美元）", color="#632ca6")
ax.tick_params(axis="y", labelcolor="#632ca6")
ax.set_title("DDOG 季度收入（柱）与同比增速（线）")

ax2 = ax.twinx()
ax2.plot(t["quarter"], t["revenue_yoy_pct"], "o--", color="#e8a33d", lw=1.8, label="同比增速")
ax2.set_ylabel("同比增速（%）", color="#e8a33d")
ax2.tick_params(axis="y", labelcolor="#e8a33d")
ax2.grid(False)
plt.xticks(rotation=90)
save(fig, "01_target_revenue")
plt.show()

print(t[["quarter", "revenue", "revenue_yoy_pct"]].tail(6).to_string(index=False))
"""),

(CODE, """# 其他披露 KPI：$100k 客户数 / billings / RPO
d = tgt.dropna(subset=["customers_100k"]).tail(14)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

axes[0].plot(d["quarter"], d["customers_100k"], "o-", color="#632ca6")
axes[0].set_title("ARR ≥ $100k 客户数")
axes[0].set_ylabel("客户数")
axes[0].tick_params(axis="x", rotation=90)

b = tgt.dropna(subset=["billings"]).tail(14)
axes[1].bar(b["quarter"], b["billings"] / 1e6, color="#e8a33d", alpha=0.85)
axes[1].set_title("Billings（收入 + 递延收入变动）")
axes[1].set_ylabel("百万美元")
axes[1].tick_params(axis="x", rotation=90)

r = tgt.dropna(subset=["rpo"]).tail(14)
axes[2].bar(r["quarter"], r["rpo"] / 1e6, color="#1f9d8f", alpha=0.85)
axes[2].set_title("RPO（剩余履约义务）")
axes[2].set_ylabel("百万美元")
axes[2].tick_params(axis="x", rotation=90)

fig.suptitle("DDOG 其他已披露 KPI", y=1.03)
save(fig, "02_target_kpis")
plt.show()
"""),

(MD, """---
## ② S1：开发者插桩流量（npm SDK 下载量）

**经济逻辑**：DDOG 按主机数 / 容器数 / 数据摄入量计费，所以"被插桩的工作负载量"是收入的物理原料；
而每次 CI/CD 部署都会拉取 SDK，因此下载量是**客户侧部署频率**的代理。

**看图要点**：注意个别 npm 包的下载量低到看不见——因为量级差异很大，右图用对数轴。
"""),

(CODE, """ow = load("observations_wide")
ow["date"] = pd.to_datetime(ow["date"])
npm_cols = [c for c in ow.columns if c.startswith("s1.npm.")]
ow["npm_total"] = ow[npm_cols].sum(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))

# 左：npm 合计（周度）
wk = ow.set_index("date")["npm_total"].resample("W").sum()
ax1.plot(wk.index, wk.values, color="#632ca6", lw=1.2)
ax1.set_title("S1 · npm SDK 周下载量（合计，2016 至今）")
ax1.set_ylabel("下载量 / 周")

# 右：分包的周下载量（对数轴，否则小包看不见）
w2 = ow.set_index("date")[npm_cols].resample("W").sum()
w2.columns = [c.replace("s1.npm.", "").replace(".downloads", "") for c in w2.columns]
for c in w2.columns:
    ax2.plot(w2.index, w2[c], lw=1.1, label=c)
ax2.set_yscale("log")
ax2.set_title("S1 · 各 SDK 周下载量（对数轴）")
ax2.set_ylabel("下载量 / 周（log）")
ax2.legend(fontsize=7, ncol=2)

save(fig, "03_s1_npm_downloads")
plt.show()

print("数据区间:", ow["date"].min().date(), "→", ow["date"].max().date(), f"（{len(ow):,} 天）")
print("各包累计下载量（百万）:")
print((ow[npm_cols].sum() / 1e6).round(1).rename(lambda s: s.replace("s1.npm.", "").replace(".downloads", "")).to_string())
"""),

(CODE, """# S1 季度同比 vs 收入同比（统一面板里已经算好）
up = load("unified_panel")
up = up.dropna(subset=["y"])

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(up["quarter"], up["y"], "s--", color="#e8a33d", lw=1.8, label="DDOG 收入同比")
ax.plot(up["quarter"], up["npm_yoy"], "o-", color="#632ca6", lw=1.5, label="npm 下载量同比")
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("同比（%）")
ax.set_title("S1 插桩流量同比 vs DDOG 收入同比")
ax.legend()
plt.xticks(rotation=90)
save(fig, "04_s1_vs_revenue")
plt.show()

print("⚠️ 注意：两条线形状相似，但这是**趋势共同上行造成的伪相关**。")
print("   一阶差分后的相关只有 %.2f（见第 ⑥ 节），走步回测也证明它没有预测增量。"
      % up["npm_yoy"].diff().corr(up["y"].diff()))
"""),

(MD, """---
## ③ S2：招聘与人力

三个不同频率、不同口径的数据源拼在一起：

- **Greenhouse 在招岗位**（日频，实时）：只能从采集当天起往前累积 → 目前只有 1 个快照
- **DOL H-1B/LCA**（季频，可回溯）：H-1B 签证申请件数，代理"招聘强度"
- **10-K 员工总数**（年频）：公司披露的真实人力水平
"""),

(CODE, """jobs = load("s2_jobs_latest")
print("在招岗位快照：", len(jobs), "个岗位")
print("快照日期取自 s2_hiring_daily：")
print(load("s2_hiring_daily")["date"].max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fn = jobs["function"].value_counts().sort_values()
ax1.barh(fn.index, fn.values, color="#632ca6", alpha=0.85)
ax1.set_title(f"S2 · 在招岗位按职能（合计 {len(jobs)} 个）")
ax1.set_xlabel("岗位数")

geo = jobs["geography"].value_counts().sort_values()
ax2.barh(geo.index, geo.values, color="#e8a33d", alpha=0.9)
ax2.set_title("S2 · 在招岗位按地区")
ax2.set_xlabel("岗位数")

save(fig, "05_s2_jobs")
plt.show()
"""),

(CODE, """dol = load("s2_dol_lca_quarterly")
hc = load("s2_headcount_annual").dropna(subset=["employees"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))

ax1.bar(dol["quarter"], dol["lca_cases"], color="#632ca6", alpha=0.85, label="全部申请")
ax1.bar(dol["quarter"], dol["lca_engineering_cases"], color="#a06cd5", alpha=0.95, label="工程岗")
ax1.set_title("S2 历史 · Datadog H-1B(LCA) 季度申请件数")
ax1.set_ylabel("件数")
ax1.legend(fontsize=8)
ax1.tick_params(axis="x", rotation=90)

ax2.bar(hc["fiscal_year"].astype(str), hc["employees"], color="#632ca6", alpha=0.85, label="员工总数")
if "employees_rnd" in hc.columns:
    ax2.bar(hc["fiscal_year"].astype(str), hc["employees_rnd"], color="#e8a33d", alpha=0.95, label="研发")
    for _, r in hc.iterrows():
        ax2.text(str(r["fiscal_year"]), r["employees"], f"{r['employees']:,.0f}",
                 ha="center", va="bottom", fontsize=7)
ax2.set_title("S2 历史 · 10-K 员工总数（年度）")
ax2.set_ylabel("人数")
ax2.legend(fontsize=8)

save(fig, "06_s2_history")
plt.show()

print("⚠️ H-1B 申请量只占公司实际年增员的 4.4%~14.8%，它测的是「签证批次」而不是业务动能，")
print("   与收入增速的相关性在 ±0.10 以内 —— 没有预测力。")
"""),

(MD, """---
## ④ 宏观基本面（ALFRED 时点口径）

这 16 个序列全部通过 **ALFRED vintage 接口**取值：每个季度只使用"该季财报日前一天"已发布的数值，
所以不含修订偏差、也不含前视。

下面挑 4 个最有经济含义的：核心资本开支订单（企业 IT 预算的上游）、金融条件、实际利率、美元。
"""),

(CODE, """macros = [
    ("NEWORDER_yoy", "核心资本品新订单 同比（%）", "企业资本开支意愿"),
    ("NFCI_lvl", "芝加哥联储金融条件指数（>0=紧）", "风险偏好/信用环境"),
    ("DFII10_lvl", "10年期 TIPS 实际收益率（%）", "真实资金成本"),
    ("DTWEXBGS_yoy", "美元贸易加权指数 同比（%）", "汇率逆风"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (col, title, sub) in zip(axes.ravel(), macros):
    d = up.dropna(subset=[col])
    ax.plot(d["quarter"], d[col], "o-", color="#1f9d8f", lw=1.5, label=col)
    ax.set_title(f"{title}\\n({sub})", fontsize=10)
    ax.tick_params(axis="x", rotation=90)
fig.suptitle("宏观因子（每季取财报日前一天的 ALFRED vintage 值）", y=1.0)
save(fig, "07_macro_factors")
plt.show()
"""),

(MD, """---
## ⑤ 云厂商景气（AMZN + MSFT + GOOGL）

DDOG 的收入是云工作量的衍生需求，而三家超大规模云厂商的财报**比 DDOG 早 1–2 周**发布
（已逐条验证 42/42 个公司-季度都早于 DDOG），因此这是**合法且及时的同期确认信号**。

⚠️ 注意右图 capex：亚马逊的资本开支口径在 FY2023 变更过，那一段的同比跳变可能是**口径断裂**而非真实加速。
"""),

(CODE, """cl = load("cloud_complex_panel")
cl = cl[cl["quarter"] >= "2019Q1"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))

ax1.plot(cl["quarter"], cl["revenue_yoy_pct"], "o-", color="#632ca6", lw=1.6)
ax1.set_title("三大云厂商 收入合计 同比")
ax1.set_ylabel("同比（%）")
ax1.tick_params(axis="x", rotation=90)

ax2.plot(cl["quarter"], cl["capex_yoy_pct"], "o-", color="#e8a33d", lw=1.6)
ax2.axvline(cl["quarter"].tolist().index("2023Q1"), color="red", ls=":", lw=1.5)
ax2.text(cl["quarter"].tolist().index("2023Q1"), ax2.get_ylim()[1] * 0.9,
         " AMZN 口径变更", color="red", fontsize=8)
ax2.set_title("三大云厂商 资本开支合计 同比")
ax2.set_ylabel("同比（%）")
ax2.tick_params(axis="x", rotation=90)

save(fig, "08_cloud_factor")
plt.show()
"""),

(MD, """---
## ⑥ 各信号与收入的关系

把候选信号和 DDOG 收入同比算相关系数。**关键区分**：

- **level**：直接对水平值求相关 → 容易拿到很高的数，但往往是"两条线都在涨"的**伪相关**
- **diff**：对一阶差分求相关 → 剔除共同趋势后，才接近真实关系

再补一张**领先-滞后**图：`corr(信号_{t-k}, 收入增速_t)`，k>0 表示信号领先 k 个季度。
"""),

(CODE, """cands = ["npm_yoy", "dol_lca_cases", "hc_employees", "NEWORDER_yoy", "NFCI_lvl",
         "DFII10_lvl", "DTWEXBGS_yoy", "cloud_rev_yoy", "cloud_capex_yoy"]
cands = [c for c in cands if c in up.columns]

rows = []
for c in cands:
    j = up[["y", c]].dropna()
    rows.append({"信号": c,
                 "level 相关": j["y"].corr(j[c]),
                 "diff 相关": j["y"].diff().corr(j[c].diff()),
                 "n": len(j)})
cor = pd.DataFrame(rows).sort_values("level 相关")

fig, ax = plt.subplots(figsize=(9, 4.6))
ypos = np.arange(len(cor))
ax.barh(ypos - 0.2, cor["level 相关"], height=0.38, color="#632ca6", alpha=0.85, label="level（易伪相关）")
ax.barh(ypos + 0.2, cor["diff 相关"], height=0.38, color="#e8a33d", alpha=0.95, label="diff（去趋势后）")
ax.set_yticks(ypos)
ax.set_yticklabels(cor["信号"])
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("与 DDOG 收入同比的相关系数")
ax.set_title("候选信号与收入增速的相关性：level vs diff")
ax.legend(fontsize=8)
save(fig, "09_correlation")
plt.show()

print(cor.round(3).to_string(index=False))
print("\\n→ npm_yoy 的 level 相关很高，但 diff 相关接近 0：典型的趋势伪相关。")
"""),

(CODE, """# 领先-滞后互相关：corr(信号_{t-k}, 收入增速_t)
signals = [c for c in ["npm_yoy", "NEWORDER_yoy", "cloud_rev_yoy", "NFCI_lvl"] if c in up.columns]

fig, ax = plt.subplots(figsize=(9, 4.4))
for c in signals:
    xs, ys = [], []
    for k in range(0, 6):
        j = pd.concat([up[c].shift(k), up["y"]], axis=1).dropna()
        xs.append(k)
        ys.append(j.corr().iloc[0, 1])
    ax.plot(xs, ys, "o-", lw=1.6, label=c)

ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("k（信号领先收入的季度数）")
ax.set_ylabel("相关系数")
ax.set_title("领先-滞后互相关 corr(信号_{t-k}, 收入增速_t)")
ax.legend(fontsize=8)
save(fig, "10_leadlag")
plt.show()

print("注意：k=0 表示同期。所有曲线随 k 增大而单调下降（而不是在某个 k>0 处出现峰值），")
print("说明这些信号并没有真正的「领先」结构 —— 高点都出现在同期或更早。")
"""),

(MD, """---
## ⑦ 结论与必须说明的局限

### 结论

**把 S1 + S2 + 32 个宏观因子 + 云厂商景气放进统一的走步回测后，没有任何一个信号能在统计上
可靠地击败朴素基准（"上季增速"外推）。** 详见 `docs/model_findings.md`。

| 对比 | naive（上季增速） | 最好的信号模型 |
|---|---|---|
| MAPE（增速口径） | **4.15%** | 6.51%（cloud_rev_yoy） |
| 方向命中率 | 不做方向判断 | 84.6%（vs 基准率 53.8%，p=0.022） |

**唯一值得继续追的方向**是 `cloud_rev_yoy`：它的**方向判断**（本季比上季加速还是减速）边际显著，
但幅度误差更大。

### 局限（写报告时不要藏）

1. **样本量是根本约束**：样本外只有 12–13 个季度。这个检出力下，"未发现显著改善"**不等于**"不存在关系"。
2. **朴素基准极强**：DDOG 增速高度自相关（长期在 25%–36% 区间缓慢移动），"上季增速"本身就是很强的预测器。
3. **多重比较**：本项目共检验约 160 个「因子 × 窗口 × 口径」组合，出现几个 p<0.05 完全在预期内。
4. **方向命中率必须对基准率检验**：窗口内若 7/12 是加速，无脑喊"加速"就有 58.3%，
   所以报告命中率时**必须同时给出基准率**。
5. **合规**：全部数据来自公开 API / 政府公开数据 / 开源仓库。`*.datadoghq.com` 被本地黑名单
   硬性拒绝（Datadog 的 Acceptable Use Policy 禁止抓取其站点，而本作业正是 Datadog 出的）。
"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        if kind == MD:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": src.splitlines(keepends=True)})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(nb: dict) -> dict:
    """逐 cell 执行，把 stdout 和 matplotlib 图形捕获成 notebook 输出。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict = {"__name__": "__main__"}
    n_ok = n_fail = 0
    idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        idx += 1
        src = "".join(cell["source"])
        plt.close("all")
        buf = io.StringIO()
        outs, err = [], None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()

        text = buf.getvalue()
        if text:
            outs.append({"output_type": "stream", "name": "stdout",
                         "text": text.splitlines(keepends=True)})
        for num in plt.get_fignums():
            b = io.BytesIO()
            plt.figure(num).savefig(b, format="png", dpi=110, bbox_inches="tight")
            outs.append({"output_type": "display_data",
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode()},
                         "metadata": {}})
        if err:
            n_fail += 1
            print(f"  ✗ cell {idx} 失败:\n{err.splitlines()[-1]}")
            outs.append({"output_type": "error", "ename": "Error",
                         "evalue": err.splitlines()[-1], "traceback": err.splitlines()})
        else:
            n_ok += 1
        cell["outputs"] = outs
        cell["execution_count"] = idx

    print(f"执行完成：{n_ok} 个 code cell 成功，{n_fail} 个失败")
    return nb


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    print(f"构建 notebook：{len([c for c in nb['cells'] if c['cell_type'] == 'code'])} 个代码单元")
    nb = execute(nb)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写出：", NB_PATH.relative_to(ROOT))
    print("图片目录：", (ROOT / "reports" / "figures" / "notebook").relative_to(ROOT))


if __name__ == "__main__":
    main()
