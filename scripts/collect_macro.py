"""宏观环境因子采集器 —— 因子分析所需的"大环境"变量（美国 GDP、利率、金融条件等）。

为什么需要它
------------
DDOG 收入 ≈ 客户云工作负载 × 单价，而客户工作负载受**大环境**驱动
（GDP/工业生产/企业资本开支订单景气、利率与金融条件松紧、通胀与美元）。
把"公司自身的另类数据信号"和"宏观环境因子"放进同一个（简单）模型里做时序预测，
就是本科计量里的**因子分析 / 因子增强预测**思路：先把共同因子（宏观环境）剥离出来，
再看公司私有信号是否还有增量解释力。

三个关键设计（都要写进报告的方法论）
-----------------------------------
1. **只采公共领域序列**：BEA / BLS / Census / 美联储体系自有指数。
   FRED 上 VIX(CBOE)、Nasdaq、Moody's Baa、密歇根消费者信心等序列带**第三方版权声明**，
   用于非个人用途需另行取得数据方许可 —— 本作业采用最保守口径，直接不采。
   FRED 条款要求注明来源并保留版权声明，且禁止用于"开发或训练任何软件/机器学习系统"，
   本项目**不涉及模型训练**，仅做研究性统计建模，且按 1.2 秒/主机限速。

2. **双端点：最新口径 + 时点(vintage)口径**
   - `fredgraph.csv`   → 最新修订值（描述性看板用）
   - `alfredgraph.csv` → **给定 as-of 日期只返回当时已发布的数据**
     这是本次方法论的核心：用最新修订值回测会发生 **revision bias**
     （后来的修订把"当时不可能知道的信息"混进了特征里）。ALFRED 让我们能用
     "DDOG 发财报前一天真实能看到的那版宏观数据"来做特征，从根本上消除前视偏差。

3. **发布滞后按频率保守假设**：日频 +1 天、周频 +10 天、月频 +45 天、季频 +45 天。
   例：2026-06 的 INDPRO 在 2026-08-05 的 vintage 里已可得（实际 7 月中旬发布），
   但 2026-07 的还看不到。宁可保守，不能乐观。

产出
----
  data/raw/fred/<日期>/<SERIES>_latest.json        每个序列的最新口径原始快照
  data/raw/alfred/<日期>/asof_<YYYY-MM-DD>.json     每个"时点"下全部序列的原始快照
  data/processed/macro_factors_long.csv            规范长表（并入 observations）
  data/processed/macro_panel_asof.csv              **时点对齐季度面板（建模用）**
  data/processed/macro_catalog.csv                 因子目录（含来源/版权/经济逻辑）

运行
----
  export no_proxy="*" NO_PROXY="*"
  .venv/bin/python scripts/collect_macro.py                 # 全量（含全部时点，约 8-10 分钟）
  .venv/bin/python scripts/collect_macro.py --latest-only    # 只更新最新口径
  .venv/bin/python scripts/collect_macro.py --rebuild-only    # 不联网，离线重算面板
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, RAW_DIR, fetch, save_raw, write_table  # noqa: E402

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2010-01-01"
ALFRED_CSV = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv"
              "?id={sid}&vintage_date={vintage}&cosd=2010-01-01")
WB_API = ("https://api.worldbank.org/v2/country/{iso}/indicator/NY.GDP.MKTP.KD.ZG"
          "?format=json&per_page=100&date=2000:2026")

# 发布滞后（天）：一个观测日的数据，要过多久才真的"看得见"
LAG_DAYS = {"D": 1, "W": 10, "M": 45, "Q": 45}

SOURCE_DAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# 因子目录：block 是"因子大类"，对应因子分析的四个维度
# ---------------------------------------------------------------------------
CATALOG: list[dict] = [
    # ── ① 宏观需求（大环境总量） ──────────────────────────────────────────
    dict(id="GDPC1", freq="Q", unit="bn USD (2017$)", block="A 宏观需求",
         owner="U.S. BEA (public domain)",
         name="美国实际 GDP（年化）",
         logic="总量的最终需求；企业 IT 预算是其滞后函数"),
    dict(id="A191RL1Q225SBEA", freq="Q", unit="% q/q SAAR", block="A 宏观需求",
         owner="U.S. BEA (public domain)",
         name="美国实际 GDP 环比折年增速",
         logic="媒体口径的\"美国 GDP 增速\"，市场对宏观强弱的直觉锚"),
    dict(id="INDPRO", freq="M", unit="index 2017=100", block="A 宏观需求",
         owner="Federal Reserve Board (public)",
         name="工业生产指数",
         logic="月频、比 GDP 早且不滞后；衡量实体活动景气"),
    dict(id="PAYEMS", freq="M", unit="thousand persons", block="A 宏观需求",
         owner="U.S. BLS (public domain)",
         name="非农就业人数",
         logic="企业扩张/收缩的先行确认，与 SaaS 席位扩张相关"),
    dict(id="NEWORDER", freq="M", unit="mn USD", block="A 宏观需求",
         owner="U.S. Census Bureau (public domain)",
         name="非国防资本品新订单（除飞机，核心资本开支）",
         logic="**企业资本开支意愿的最直接代理** —— 云/可观测性支出的上游"),
    # ── ② 利率与流动性（贴现率/融资成本环境） ────────────────────────────
    dict(id="DGS10", freq="D", unit="%", block="B 利率流动性",
         owner="U.S. Treasury via Fed H.15 (public)",
         name="10 年期美债收益率",
         logic="贴现率与风险资产估值之锚；高利率压制 IT 预算与 SaaS 估值"),
    dict(id="DFII10", freq="D", unit="%", block="B 利率流动性",
         owner="U.S. Treasury via Fed (public)",
         name="10 年期 TIPS 实际收益率",
         logic="剔除通胀预期的真实资金成本，比名义利率更干净的贴现率"), 
    dict(id="FEDFUNDS", freq="M", unit="%", block="B 利率流动性",
         owner="Federal Reserve Board (public)",
         name="联邦基金有效利率",
         logic="货币政策松紧，影响风险偏好与企业融资"),
    dict(id="T10Y2Y", freq="D", unit="pp", block="B 利率流动性",
         owner="Federal Reserve Board (public)",
         name="10Y-2Y 期限利差",
         logic="市场对增长/衰退的定价；倒挂是衰退领先指标"),
    # ── ③ 金融条件与风险偏好 ─────────────────────────────────────────────
    dict(id="NFCI", freq="W", unit="index (0=均值)", block="C 金融条件",
         owner="Federal Reserve Bank of Chicago (public)",
         name="芝加哥联储全国金融条件指数",
         logic="风险偏好/信用环境的综合度量；正=紧，负=松"),
    # 注：曾试图并列 STLFSI4（圣路易斯联储金融压力指数），但实测 ALFRED 在 2023 年之前的
    # vintage 取不到该序列（STLFSI4 是 2023 年启用的新版本，历史回溯不覆盖 2018-2022 的发布时点），
    # 会造成整整 16 个季度缺失 → 已剔除；金融条件维度由 NFCI 单独代表（其历史完整）。
    # ── ④ 通胀与汇率 ─────────────────────────────────────────────────────
    dict(id="CPIAUCSL", freq="M", unit="index 1982-84=100", block="D 通胀汇率",
         owner="U.S. BLS (public domain)",
         name="CPI 城市消费者物价指数",
         logic="成本与定价环境；影响实际 IT 预算与美元名义收入"),
    dict(id="DTWEXBGS", freq="D", unit="index 2006=100", block="D 通胀汇率",
         owner="Federal Reserve Board (public)",
         name="美元名义广义贸易加权指数",
         logic="DDOG 有海外收入；美元走强压低美元计价收入（汇率逆风）"),
    # ── ⑤⑥ 补充：半导体/电子生产与景气调查（均为美联储体系序列）──────────
    dict(id="IPG3344S", freq="M", unit="index 2017=100", block="E 行业景气",
         owner="Federal Reserve Board G.17 (public)",
         name="计算机与电子产品工业生产指数",
         logic="云/AI 硬件周期的最上游；硬件出货→软件装机→可观测性用量"),
    dict(id="WPU38110101", freq="M", unit="index Dec 2006=100", block="E 行业景气",
         owner="U.S. BLS (public domain) via FRED",
         name="数据处理、托管与 IT 基础设施服务 PPI",
         logic="云服务价格与行业商业化环境；用于观察价格变化，不直接代表需求量"),
    dict(id="CFNAI", freq="M", unit="index (0=趋势增长)", block="A 宏观需求",
         owner="Federal Reserve Bank of Chicago (public)",
         name="芝加哥联储全国活动指数",
         logic="85 个指标合成的月度景气总指数，比 GDP 更早、噪音更低"),
    dict(id="GACDFSA066MSFRBPHI", freq="M", unit="index (>0=扩张)", block="F 景气调查",
         owner="Federal Reserve Bank of Philadelphia (public)",
         name="费城联储制造业景气指数",
         logic="ISM PMI 因版权 2016 年已从 FRED 下架，地区联储调查是免费替代品"),
    dict(id="GACDISA066MSFRBNY", freq="M", unit="index (>0=扩张)", block="F 景气调查",
         owner="Federal Reserve Bank of New York (public)",
         name="纽约联储制造业景气指数",
         logic="同上的第二家地区联储调查，用于交叉验证景气拐点"),
]

# 说明：ICE BofA 高收益信用利差（BAMLH0A0HYM2）与投资级（BAMLC0A0CM）是常见选择，
# 但本机实测该两序列在 FRED 的 CSV 端点只返回约 3 年历史（796 行，起于 2023-09），
# 无法覆盖 2018–2026 的回测窗口；且版权属 ICE BofA（非公共领域）。
# 故改用美联储自有的 NFCI / STLFSI4 代表"金融条件/信用环境"，结论可复现且合规。

WB_SERIES = [
    dict(iso="USA", key="macro.wb.USA.gdp_growth", name="美国 GDP 增速（世界银行口径，年度）"),
    dict(iso="WLD", key="macro.wb.WLD.gdp_growth", name="全球 GDP 增速（世界银行口径，年度）"),
]

LEGAL = {
    "fred": ("FRED public CSV download endpoint (fredgraph.csv), Federal Reserve Bank of "
             "St. Louis - latest-vintage values; attribution to FRED given; no ML training use"),
    "alfred": ("ALFRED archival CSV endpoint (alfredgraph.csv), Federal Reserve Bank of "
               "St. Louis - point-in-time (as-of vintage) values, used to avoid revision bias"),
    "worldbank": "World Bank Open Data API (CC BY 4.0)",
}

# 因子系列在规范长表里用的前缀
PREFIX = "macro.fred."


# ---------------------------------------------------------------------------
# 解析 / 抓取
# ---------------------------------------------------------------------------
def parse_fred_csv(text: str) -> list[list]:
    """fredgraph/alfredgraph CSV → [[date, value|null], ...]（第二列名带 vintage 后缀）。"""
    out: list[list] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if not d:
            continue
        try:
            val = float(v) if v not in ("", ".", "NaN", "null") else None
        except ValueError:
            val = None
        out.append([d, val])
    return out


def fetch_latest(entry: dict) -> list[list]:
    url = FRED_CSV.format(sid=entry["id"])
    r = fetch(url, source="fred", note=f"FRED latest {entry['id']}")
    rows = parse_fred_csv(r.text)
    save_raw("fred", f"{entry['id']}_latest",
             {"series_id": entry["id"], "name": entry["name"], "frequency": entry["freq"],
              "unit": entry["unit"], "owner": entry["owner"], "observations": rows},
             url=url, note=f"FRED latest-vintage {entry['id']} ({entry['name']})")
    return rows


def _vintage_path(vintage: date) -> Path:
    """已存在则复用（可增量补齐新序列），否则放今天的目录。"""
    hits = sorted((RAW_DIR / "alfred").glob(f"*/asof_{vintage.isoformat()}.json"))
    return hits[-1] if hits else (RAW_DIR / "alfred" / SOURCE_DAY / f"asof_{vintage.isoformat()}.json")


def fetch_vintage(vintage: date, entries: list[dict]) -> dict:
    """抓某个 as-of 日期下全部序列的时点值，合并成一个原始快照。

    幂等：若快照已存在且已含全部序列，直接跳过（便于后续往目录里加新因子）。
    """
    p = _vintage_path(vintage)
    existing: dict = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))["payload"].get("series", {})
        except Exception:
            existing = {}
    missing = [e for e in entries
               if not existing.get(e["id"], {}).get("observations")]
    if not missing:
        print("      (快照已完整，跳过)")
        return existing

    payload = dict(existing)
    for e in missing:
        url = ALFRED_CSV.format(sid=e["id"], vintage=vintage.isoformat())
        try:
            r = fetch(url, source="alfred", note=f"ALFRED as-of {vintage} {e['id']}")
            payload[e["id"]] = {"url": url, "frequency": e["freq"],
                                "observations": parse_fred_csv(r.text)}
        except Exception as exc:                       # 单序列失败不拖垮整个 vintage
            print(f"    ! {vintage} {e['id']}: {str(exc)[:80]}")
            payload[e["id"]] = {"url": url, "frequency": e["freq"], "observations": [],
                                "error": str(exc)[:200]}
    save_raw("alfred", f"asof_{vintage.isoformat()}",
             {"vintage": vintage.isoformat(), "series": payload},
             url=f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?vintage_date={vintage}",
             note=f"ALFRED point-in-time snapshot as of {vintage}",
             day=p.parent.name)
    return payload


def fetch_worldbank() -> pd.DataFrame:
    recs = []
    for s in WB_SERIES:
        url = WB_API.format(iso=s["iso"])
        r = fetch(url, source="worldbank", note=f"World Bank GDP growth {s['iso']}")
        js = r.json()
        save_raw("worldbank", f"gdp_growth_{s['iso']}", js, url=url,
                 note=f"World Bank annual real GDP growth, {s['iso']}")
        for row in (js[1] if len(js) > 1 and js[1] else []):
            if row.get("value") is None:
                continue
            recs.append({"date": f"{row['date']}-12-31", "series_id": s["key"],
                         "value": float(row["value"]), "unit": "% yoy", "freq": "Y",
                         "source": "worldbank", "legal_basis": LEGAL["worldbank"],
                         "description": s["name"]})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 时点面板构建
# ---------------------------------------------------------------------------
def _qkey(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _q_prev_year(q: str) -> str:
    return f"{int(q[:4]) - 1}Q{q[-1]}"


def agg_by_quarter(rows: list[list], asof: date, lag_days: int) -> dict[str, dict]:
    """把观测聚合成日历季度，**只使用 as-of 时点已发布的数据**。

    同时记录"月份位置"（季度内第 1/2/3 个月）的均值，用于计算**同窗口同比**：
    只比两边都真实可见的月份，避免"本期 2 个月 vs 去年 3 个月"造成的机械偏差。
    """
    out: dict[str, dict] = {}
    for d_str, v in rows:
        if v is None:
            continue
        d = date.fromisoformat(d_str)
        if d + timedelta(days=lag_days) > asof:
            continue
        q = _qkey(d)
        off = ((d.month - 1) % 3) + 1
        rec = out.setdefault(q, {"vals": [], "off": {}, "last": d})
        rec["vals"].append(v)
        rec["off"].setdefault(off, []).append(v)
        if d > rec["last"]:
            rec["last"] = d
    for rec in out.values():
        rec["mean"] = float(np.mean(rec["vals"]))
        rec["off_mean"] = {k: float(np.mean(v)) for k, v in rec["off"].items()}
    return out


def build_panel(vintages: list[tuple[str, date]], fetch_day: str | None = None) -> pd.DataFrame:
    """从磁盘上的原始快照重算时点面板（离线可复现，不用重新联网）。"""
    by_id = {e["id"]: e for e in CATALOG}
    rows: list[dict] = []
    # 修复（2026-09-14 审计发现）：原实现只在**单个**快照目录里找 vintage 文件。
    # 但 _vintage_path() 会把新 vintage 写进"今天"的目录，因此一旦跨日重跑，
    # 这里只能找到当天新抓的 1 个文件，其余 31 个 vintage 各打印一行提示后被跳过，
    # 却**仍然写出** macro_panel_asof.csv —— 505 行会被静默截断成十几行且不报错。
    # 现在改为索引**全部** alfred 快照目录（同名文件取最新目录里的），
    # 并在快照不全时直接拒绝写出，避免下游拿到残缺面板。
    alfred_dirs = sorted((RAW_DIR / "alfred").glob("*")) if (RAW_DIR / "alfred").exists() else []
    if not alfred_dirs:
        raise SystemExit("找不到 ALFRED 快照，请先运行不带 --rebuild-only 的采集")
    index: dict[str, Path] = {}
    for d in alfred_dirs:
        for p in sorted(d.glob("asof_*.json")):
            index[p.name] = p            # 目录名有序，后覆盖前 = 取最新目录中的版本
    missing = [v for _, v in vintages if f"asof_{v.isoformat()}.json" not in index]
    if missing:
        raise SystemExit(
            f"缺少 {len(missing)}/{len(vintages)} 个 vintage 快照，拒绝写出被截断的面板。\n"
            f"  缺失示例：{[v.isoformat() for v in missing[:5]]}\n"
            f"  已索引 {len(index)} 个快照，目录：{[d.name for d in alfred_dirs]}\n"
            f"  请先运行采集补齐这些 vintage 再重建面板。"
        )

    for quarter, vintage in vintages:
        p = index[f"asof_{vintage.isoformat()}.json"]
        snap = json.loads(p.read_text(encoding="utf-8"))["payload"]
        for sid, blob in snap.get("series", {}).items():
            e = by_id.get(sid)
            if e is None:          # 快照里可能有已从目录中剔除的序列（如 STLFSI4），跳过
                continue
            agg = agg_by_quarter(blob.get("observations", []), vintage, LAG_DAYS[e["freq"]])
            cur, prev = agg.get(quarter), agg.get(_q_prev_year(quarter))
            if not cur:
                continue
            yoy = None
            if prev:
                common = set(cur["off_mean"]) & set(prev["off_mean"])
                if common:
                    a = np.mean([cur["off_mean"][k] for k in common])
                    b = np.mean([prev["off_mean"][k] for k in common])
                    if b != 0:
                        yoy = (a / b - 1) * 100
            rows.append({
                "quarter": quarter, "vintage": vintage.isoformat(),
                "series_key": PREFIX + sid, "block": e["block"], "factor": e["name"],
                "freq": e["freq"], "unit": e["unit"], "owner": e["owner"],
                "value": cur["mean"], "value_yoy_pct": yoy,
                "coverage_months": len(cur["off"]), "last_obs_date": cur["last"].isoformat(),
                "economic_logic": e["logic"], "legal_basis": LEGAL["alfred"],
            })
    return pd.DataFrame(rows)


def ddog_vintages() -> list[tuple[str, date]]:
    """每个 DDOG 财季对应的 as-of 时点 = 该季财报发布日前一天。

    用 8-K 新闻稿日（= 财报日，盘后发布）减 1 天，确保特征里不含当天盘后信息。
    """
    tgt = pd.read_parquet(PROCESSED_DIR / "ddog_target_quarterly.parquet")
    out = []
    for _, r in tgt.iterrows():
        pe = pd.Timestamp(r["period_end"])
        fd = pd.Timestamp(r["filing_date"]) if pd.notna(r.get("filing_date")) else pe + pd.Timedelta(days=40)
        out.append((r["quarter"], (fd - pd.Timedelta(days=1)).date()))
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest-only", action="store_true", help="只更新最新口径，不抓时点")
    ap.add_argument("--rebuild-only", action="store_true", help="不联网，仅用已有快照重算面板")
    args = ap.parse_args()

    print("== 宏观环境因子采集 ==")
    print(f"   目录：{len(CATALOG)} 个 FRED/ALFRED 序列 + {len(WB_SERIES)} 个世界银行序列")
    print("   口径：仅公共领域（BEA/BLS/Census/美联储体系），避开带第三方版权声明的序列")

    long_parts: list[pd.DataFrame] = []
    today_v = date.today() - timedelta(days=1)
    # 最后一个 vintage = "现在"（用于给尚未披露的当前季度做实时 nowcast）
    vintages = ddog_vintages() + [(_qkey(today_v), today_v)]

    if not args.rebuild_only:
        print(f"\n-- 最新口径：{len(CATALOG)} 个序列 --")
        for e in CATALOG:
            rows = fetch_latest(e)
            ok = [r for r in rows if r[1] is not None]
            print(f"   {e['id']:<18} {len(ok):>6} 个观测  {ok[0][0] if ok else '-'} → {ok[-1][0] if ok else '-'}")

        if not args.latest_only:
            print(f"\n-- 时点口径：{len(vintages)} 个 vintage × {len(CATALOG)} 序列 "
                  f"（约 {len(vintages) * len(CATALOG) * 1.2 / 60:.0f} 分钟，请耐心） --")
            for i, (q, v) in enumerate(vintages, 1):
                print(f"   [{i:>2}/{len(vintages)}] {q} as-of {v}", flush=True)
                fetch_vintage(v, CATALOG)

        print("\n-- 世界银行年度大环境（背景口径） --")
        wb = fetch_worldbank()
        print(f"   {len(wb)} 行")

    # ---- 规范长表（最新口径，并入 observations）----
    for e in CATALOG:
        p = RAW_DIR / "fred" / SOURCE_DAY / f"{e['id']}_latest.json"
        if not p.exists():
            cands = sorted((RAW_DIR / "fred").glob(f"*/{e['id']}_latest.json"))
            if not cands:
                continue
            p = cands[-1]
        obs = json.loads(p.read_text(encoding="utf-8"))["payload"]["observations"]
        df = pd.DataFrame([{"date": d, "series_id": PREFIX + e["id"], "value": v,
                            "unit": e["unit"], "freq": e["freq"], "source": "fred",
                            "legal_basis": LEGAL["fred"],
                            "description": f"[{e['block']}] {e['name']} · {e['owner']}"}
                           for d, v in obs if v is not None])
        long_parts.append(df)
    if (RawWB := list((RAW_DIR / "worldbank").glob("*/gdp_growth_*.json"))):
        wb_parts = []
        for p in RawWB:
            js = json.loads(p.read_text(encoding="utf-8"))
            payload = js.get("payload", js)
            iso = p.stem.replace("gdp_growth_", "")
            key = f"macro.wb.{iso}.gdp_growth"
            nm = next((s["name"] for s in WB_SERIES if s["iso"] == iso), key)
            for row in (payload[1] if len(payload) > 1 and payload[1] else []):
                if row.get("value") is None:
                    continue
                wb_parts.append({"date": f"{row['date']}-12-31", "series_id": key,
                                 "value": float(row["value"]), "unit": "% yoy", "freq": "Y",
                                 "source": "worldbank", "legal_basis": LEGAL["worldbank"],
                                 "description": f"[E 全球背景] {nm}"})
        if wb_parts:
            long_parts.append(pd.DataFrame(wb_parts))

    if long_parts:
        long_df = (pd.concat(long_parts, ignore_index=True)
                   .drop_duplicates(["date", "series_id"], keep="last")
                   .sort_values(["series_id", "date"]))
        print(f"\n-- 规范长表：{len(long_df)} 行 / {long_df['series_id'].nunique()} 序列 --")
        for p in write_table(long_df, "macro_factors_long"):
            print("  ->", p)

    # ---- 时点面板 ----
    print("\n-- 构建时点对齐季度面板（每季取财报日前一天的 vintage） --")
    panel = build_panel(vintages)
    if len(panel):
        for p in write_table(panel, "macro_panel_asof"):
            print("  ->", p)
        print(f"   {len(panel)} 行；季度 {panel['quarter'].nunique()} 个；"
              f"因子 {panel['series_key'].nunique()} 个")

    # ---- 因子目录 ----
    cat = pd.DataFrame([{"series_key": PREFIX + e["id"], "factor": e["name"], "block": e["block"],
                         "freq": e["freq"], "unit": e["unit"], "owner": e["owner"],
                         "economic_logic": e["logic"], "fred_url":
                         f"https://fred.stlouisfed.org/series/{e['id']}"} for e in CATALOG]
                        + [{"series_key": s["key"], "factor": s["name"], "block": "E 全球背景",
                            "freq": "Y", "unit": "% yoy", "owner": "World Bank (CC BY 4.0)",
                            "economic_logic": "全球增长的年度背景（频率过低，不进季度模型）",
                            "fred_url": ""} for s in WB_SERIES])
    for p in write_table(cat, "macro_catalog"):
        print("  ->", p)

    if len(panel):
        print("\n== 面板抽样（最近 4 个季度 × 4 个因子） ==")
        show = panel[panel["series_key"].isin(
            [PREFIX + x for x in ["GDPC1", "NEWORDER", "DFII10", "NFCI"]])]
        print(show[["quarter", "vintage", "series_key", "value", "value_yoy_pct",
                    "coverage_months", "last_obs_date"]].tail(16).to_string(index=False))


if __name__ == "__main__":
    main()
