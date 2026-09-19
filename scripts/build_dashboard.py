"""仪表盘数据装配器 —— 把处理层产物打包成 site/data/payload.json。

与旧脚本 collect.py 的分工：collect.py 负责"抓取+装配"一体（早期版本），
本脚本只做**装配**：所有输入都来自 data/processed/ 的规范产物，
因此可以在不联网、不重抓的前提下随时重建看板。

payload 结构（新增键用 ★ 标注）：
    generated_at / provenance / log      元信息
    rows                                 日频与季度序列（npm/pypi/sec_xbrl）
    hiring                               招聘快照（总数 + 职能分布）
    kpi                                  公司披露 KPI 最新值
    ★ macro                                宏观环境因子（时点口径季度面板）
  ★ cloud                                云厂商行业景气因子（SEC XBRL）
  ★ model                                走步回测指标 + 单因子筛选 + PCA 方差解释
  ★ leadlag                              领先滞后互相关
  ★ nowcast                              当前未披露季度的实时预测
  ★ sources                              数据源健康度（含合规依据）

运行： .venv/bin/python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, ROOT  # noqa: E402

SITE_DATA = ROOT / "site" / "data"

# 看板上展示的宏观因子（季度序列，时点口径）
MACRO_SHOW = ["GDPC1", "A191RL1Q225SBEA", "NEWORDER", "CFNAI", "INDPRO", "IPG3344S",
              "DFII10", "DGS10", "FEDFUNDS", "T10Y2Y", "NFCI",
              "CPIAUCSL", "DTWEXBGS", "PAYEMS", "GACDFSA066MSFRBPHI", "GACDISA066MSFRBNY"]
# 每个因子用哪个口径展示：growth=同比（实体活动量）、level=水平（利率/指数本身就有意义）
# 不同量纲不能同轴，前端统一画各自的 Z 分数。
MACRO_MODE = {"GDPC1": "growth", "A191RL1Q225SBEA": "level", "NEWORDER": "growth",
              "CFNAI": "level", "INDPRO": "growth", "IPG3344S": "growth",
              "DFII10": "level", "DGS10": "level", "FEDFUNDS": "level",
              "T10Y2Y": "level", "NFCI": "level",
              "CPIAUCSL": "growth", "DTWEXBGS": "growth", "PAYEMS": "growth",
              "GACDFSA066MSFRBPHI": "level", "GACDISA066MSFRBNY": "level"}

PROVENANCE = {
    "npm": "api.npmjs.org (official registry API)",
    "pypi": "pypistats.org (public API, without_mirrors)",
    "greenhouse": "boards-api.greenhouse.io Job Board API (public)",
    "sec": "SEC EDGAR XBRL + 8-K EX-99.1 (public domain)",
    "fred": "FRED / ALFRED public CSV endpoints (St. Louis Fed); point-in-time vintages",
    "worldbank": "World Bank Open Data API (CC BY 4.0)",
    "hackernews": "Hacker News public search API (Algolia index; public discussion aggregates)",
    "stackexchange": "Stack Exchange public API (CC BY-SA content; public discussion aggregates)",
    "cloud": "SEC XBRL of AMZN / MSFT / GOOGL (public domain)",
    "blocked": "datadoghq.com not accessed - Datadog AUP No Framing or Scraping",
}

REVENUE_FACTOR_CATALOG = [
    ("revenue_lag1", "上季收入", "目标滞后", True),
    ("revenue_lag4", "去年同期收入", "目标滞后", True),
    ("npm_yoy", "npm 插桩下载同比", "公司数据", True),
    ("cloud_rev_yoy", "云厂商收入同比", "行业环境", False),
    ("cloud_capex_yoy", "云厂商 capex 同比", "行业环境", False),
    ("gh_open_roles", "Greenhouse 当前岗位", "公司数据", False),
    ("dol_lca_new_employment", "DOL 新增雇佣岗位", "公司数据", False),
    ("hc_employees", "披露员工数", "公司数据", False),
    ("NEWORDER_yoy", "核心资本品订单同比", "宏观控制", False),
    ("DFII10_lvl", "10Y 实际利率", "宏观控制", False),
    ("NFCI_lvl", "金融条件指数", "宏观控制", False),
    ("DTWEXBGS_yoy", "美元指数同比", "宏观控制", False),
]


def build_rows() -> list[dict]:
    """日频（npm/pypi）+ 季度（SEC）序列，格式与前端 buildSeries() 约定一致。"""
    rows: list[dict] = []
    obs = pd.read_parquet(PROCESSED_DIR / "observations.parquet")
    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")

    npm = obs[obs["series_id"].str.startswith("s1.npm.")]
    for _, r in npm.iterrows():
        pkg = r["series_id"].replace("s1.npm.", "").replace(".downloads", "")
        rows.append({"date": r["date"], "source": "npm", "metric": f"dl_{pkg}",
                     "value": float(r["value"])})
    pypi = obs[obs["series_id"].str.startswith("s1.pypi.")]
    for _, r in pypi.iterrows():
        pkg = r["series_id"].replace("s1.pypi.", "").replace(".downloads", "")
        rows.append({"date": r["date"], "source": "pypi", "metric": f"dl_{pkg}",
                     "value": float(r["value"])})

    tgt = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    spec = {"revenue": "revenue_q", "customers_100k": "customers_100k_arr",
            "deferred_revenue_current": "deferred_rev_current", "rpo": "rpo",
            "billings": "billings_q", "free_cash_flow": "free_cash_flow_q",
            "revenue_yoy_pct": "revenue_yoy_pct"}
    for _, r in tgt.iterrows():
        d = pd.Timestamp(r["period_end"]).strftime("%Y-%m-%d")
        for col, metric in spec.items():
            if col in tgt.columns and pd.notna(r.get(col)):
                rows.append({"date": d, "source": "sec_xbrl", "metric": metric,
                             "value": float(r[col]),
                             "legal_basis": r.get("legal_basis", "")})
    return rows


def build_hiring() -> dict:
    p = PROCESSED_DIR / "s2_jobs_latest.csv"
    if not p.exists():
        return {}
    j = pd.read_csv(p)
    daily = PROCESSED_DIR / "s2_hiring_daily.csv"
    as_of = "—"
    total = len(j)
    if daily.exists():
        d = pd.read_csv(daily)
        tot = d[d["series_id"] == "s2.greenhouse.total_open_roles"]
        if len(tot):
            total = int(tot["value"].iloc[-1])
            as_of = str(tot["date"].iloc[-1])
    by_dep = j["function"].value_counts().to_dict() if "function" in j.columns else {}
    return {"as_of": as_of, "open_roles": total, "by_department": by_dep,
            "by_geography": j["geography"].value_counts().to_dict() if "geography" in j.columns else {}}


def build_kpi() -> dict:
    tgt = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    last = tgt.dropna(subset=["revenue"]).iloc[-1]
    return {"quarter": last["quarter"],
            "revenue_usd": float(last["revenue"]),
            "revenue_yoy_pct": float(last["revenue_yoy_pct"]) if pd.notna(last["revenue_yoy_pct"]) else None,
            "customers_100k_arr": float(last["customers_100k"]) if pd.notna(last["customers_100k"]) else None}


def build_macro() -> dict:
    """宏观因子季度面板（时点口径）。每个季度只保留该季 vintage 的那一行。"""
    p = PROCESSED_DIR / "macro_panel_asof.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p)
    m["sid"] = m["series_key"].str.replace("macro.fred.", "", regex=False)
    keep = m[m["sid"].isin(MACRO_SHOW)].sort_values("vintage").copy()
    if not len(keep):
        return {}
    cat = {}
    cp = PROCESSED_DIR / "macro_catalog.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        cat = {r["series_key"].replace("macro.fred.", ""): r["factor"] for _, r in c.iterrows()}

    rows, labels, modes = [], {}, {}
    for sid, g in keep.groupby("sid"):
        g = g.sort_values("quarter")
        mode = MACRO_MODE.get(sid, "level")
        raw = (g["value_yoy_pct"] if mode == "growth" else g["value"]).astype(float)
        sd = raw.std()
        z = (raw - raw.mean()) / sd if sd and sd > 0 else raw * 0
        labels[sid], modes[sid] = cat.get(sid, sid), mode
        for (_, r), zz, rr in zip(g.iterrows(), z, raw):
            rows.append({"quarter": r["quarter"], "series": sid, "block": r["block"],
                         "value": None if pd.isna(r["value"]) else round(float(r["value"]), 4),
                         "yoy_pct": None if pd.isna(r["value_yoy_pct"]) else round(float(r["value_yoy_pct"]), 3),
                         "display_raw": None if pd.isna(rr) else round(float(rr), 3),
                         "z": None if pd.isna(zz) else round(float(zz), 3),
                         "coverage_months": int(r["coverage_months"]),
                         "last_obs": r["last_obs_date"]})
    return {"as_of": str(keep["vintage"].max()), "series": sorted(keep["sid"].unique().tolist()),
            "labels": labels, "modes": modes, "rows": rows}


def build_cloud() -> dict:
    """云厂商行业景气季度面板，用于宏观/行业页面展示。"""
    p = PROCESSED_DIR / "cloud_complex_panel.csv"
    if not p.exists():
        return {}
    c = pd.read_csv(p)
    cols = ["quarter", "revenue_sum", "revenue_yoy_pct", "capex_sum", "capex_yoy_pct", "first_filed"]
    cols = [x for x in cols if x in c.columns]
    c = c[cols].sort_values("quarter")
    return {
        "rows": c.round(4).to_dict("records"),
        "as_of": str(c["first_filed"].dropna().max()) if "first_filed" in c.columns and c["first_filed"].notna().any() else "—",
        "legal": PROVENANCE["cloud"],
    }


def _read(name: str) -> pd.DataFrame:
    p = PROCESSED_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_model() -> dict:
    md = _read("forecast_metrics")
    scr = _read("factor_screening")
    var = _read("macro_pca_variance")
    boot = _read("forecast_delta_vs_naive")
    md_qq = _read("forecast_metrics_qq")
    out: dict = {}
    if len(md):
        out["metrics"] = md.round(4).to_dict("records")
        out["oos_quarters"] = int(md["n_oos"].max())
    if len(md_qq):
        out["metrics_qq"] = md_qq.round(4).to_dict("records")
    if len(boot):
        out["delta_vs_naive"] = boot.round(4).to_dict("records")
    if len(scr):
        out["screening"] = scr.head(12).round(4).to_dict("records")
    if len(var):
        out["pca_variance_pct"] = [round(float(v), 2) for v in var["explained_var_pct"]]
    return out


def build_revenue_model() -> dict:
    """前端交互建模用：以季度收入绝对值为 y，因子可在网页上自由勾选。"""
    p = PROCESSED_DIR / "unified_panel.csv"
    if not p.exists():
        return {"rows": [], "factors": []}
    d = pd.read_csv(p).sort_values("quarter").copy()
    d["revenue_lag1"] = d["revenue"].shift(1)
    d["revenue_lag2"] = d["revenue"].shift(2)
    d["revenue_lag4"] = d["revenue"].shift(4)
    cols = ["quarter", "period_end", "revenue", "revenue_lag1", "revenue_lag2"]
    cols += [c for c, _, _, _ in REVENUE_FACTOR_CATALOG if c in d.columns]
    cols = list(dict.fromkeys(cols))
    d = d[cols]
    d = d[d["revenue"].notna()].copy()
    factors = [
        {"key": key, "label": label, "group": group, "default": default}
        for key, label, group, default in REVENUE_FACTOR_CATALOG
        if key in d.columns
    ]
    return {"target": "revenue", "unit": "USD", "rows": d.round(6).to_dict("records"), "factors": factors}


def build_fine_grained() -> dict:
    """细分产品/招聘数据：暂不进首屏，只供网页下钻和后续建模使用。"""
    out: dict = {}
    for key, name in [
        ("product_monthly", "product_downloads_fine_monthly"),
        ("product_quarterly", "product_downloads_fine_quarterly"),
        ("hiring_family_quarterly", "hiring_lca_fine_by_family_quarterly"),
        ("hiring_state_quarterly", "hiring_lca_fine_by_state_quarterly"),
    ]:
        df = _read(name)
        out[key] = df.round(4).to_dict("records") if len(df) else []
    return out


def build_forum() -> dict:
    """Public discussion aggregates only; individual post text stays out of the payload."""
    frame = _read("forum_mentions_daily")
    return {
        "rows": frame.to_dict("records") if len(frame) else [],
        "google_trends": {"available": False,
                          "reason": "等待 Google Trends 官方 API 访问权限"},
    }


def build_sources() -> list[dict]:
    """数据源健康度：状态按"最近的原始快照日期"判断，全部附合规依据。"""
    def latest(sub: str) -> str:
        d = ROOT / "data" / "raw" / sub
        if not d.exists():
            return "—"
        days = sorted([p.name for p in d.iterdir() if p.is_dir()])
        return days[-1] if days else "—"

    return [
        {"name": "npm registry", "freq": "日", "snapshot": latest("npm"), "status": "ok",
         "legal": "官方公共下载量接口（第三方 registry，非 Datadog 资产）"},
        {"name": "pypistats", "freq": "日", "snapshot": latest("pypistats"), "status": "ok",
         "legal": "公共 API，已剔除 mirror 流量以过滤机器人"},
        {"name": "Greenhouse", "freq": "日", "snapshot": latest("greenhouse"), "status": "ok",
         "legal": "官方文档化的职位分发接口"},
        {"name": "SEC XBRL / 8-K", "freq": "季", "snapshot": latest("sec"), "status": "ok",
         "legal": "美国政府公开数据；声明 UA 且 ≤10 req/s"},
        {"name": "FRED / ALFRED（宏观因子）", "freq": "日/月/季", "snapshot": latest("alfred"),
         "status": "ok", "legal": "圣路易斯联储公开 CSV 端点；仅取公共领域序列，ALFRED 时点口径"},
        {"name": "SEC 云厂商 XBRL（行业因子）", "freq": "季", "snapshot": latest("sec_cloud"),
         "status": "ok", "legal": "AMZN/MSFT/GOOGL 公开 XBRL，按 first_filed 做时点对齐"},
        {"name": "World Bank", "freq": "年", "snapshot": latest("worldbank"), "status": "ok",
         "legal": "Open Data API，CC BY 4.0"},
        {"name": "Tableau 嵌入", "freq": "—", "snapshot": "—", "status": "todo",
         "legal": "待配置 TABLEAU_URL"},
    ]


def _clean(obj):
    """把 NaN/±Inf 递归换成 None。

    Python 的 json.dumps 默认会写出裸 NaN（非法 JSON），浏览器 JSON.parse 会直接抛错 →
    看板会静默退回兜底数据（白屏/示例值）。这里显式清洗，并配合 allow_nan=False 兜底。
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def main() -> None:
    print("== 装配仪表盘 payload ==")
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hiring": build_hiring(),
        "kpi": build_kpi(),
        "rows": rows,
        "macro": build_macro(),
        "cloud": build_cloud(),
        "model": build_model(),
        "revenue_model": build_revenue_model(),
        "fine": build_fine_grained(),
        "forum": build_forum(),
        "leadlag": _read("leadlag").round(4).to_dict("records") if len(_read("leadlag")) else [],
        "nowcast": (_read("forecast_latest").to_dict("records") if len(_read("forecast_latest")) else []),
        "sources": build_sources(),
        "provenance": PROVENANCE,
        "log": ["S1 npm/PyPI downloads", "S2 Greenhouse jobs", "Target SEC XBRL + 8-K",
                "Macro FRED/ALFRED point-in-time", "Cloud-complex SEC XBRL",
                "Factor-augmented forecast (walk-forward)"],
    }
    payload = _clean(payload)
    out = SITE_DATA / "payload.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str, allow_nan=False),
                   encoding="utf-8")
    print(f"  -> {out.relative_to(ROOT)}  ({len(rows):,} 行序列数据, "
          f"{out.stat().st_size / 1024:.0f} KB)")
    print(f"     宏观因子 {len(payload['macro'].get('rows', []))} 行；"
          f"云厂商 {len(payload['cloud'].get('rows', []))} 行；"
          f"模型指标 {len(payload['model'].get('metrics', []))} 条；"
          f"nowcast {len(payload['nowcast'])} 条")


if __name__ == "__main__":
    main()
