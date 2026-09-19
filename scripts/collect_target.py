"""目标变量采集器 —— DDOG 已披露的季度 KPI（预测的 y）。

数据源：SEC EDGAR / XBRL 公开接口（美国政府公开数据，官方允许程序化访问）。
产出：
  data/processed/ddog_target_quarterly.csv    季度面板：收入 / 递延收入 / billings / $100k 客户数
  data/raw/sec/<date>/*.json                  原始快照

说明：
  - Q1–Q3 收入取 10-Q 单季值；**Q4 由 10-K 年度收入减去前三季倒算**（10-K 不单独标记 Q4）；
  - billings = 收入 + 递延收入变动（DDOG 口径的近似）；
  - $100k 客户数从 8-K 的 EX-99.1 新闻稿正文正则提取。
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import LEGAL_BASIS, fetch, get_json, save_raw, write_table  # noqa: E402

CIK = "0001561550"
# 注意：归档文件在 www.sec.gov 上；data.sec.gov 只提供 XBRL/JSON API（实测前者 404）
BASE = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}"


_QMAP = {"03": 1, "06": 2, "09": 3, "12": 4}


def _q_label(period_end: str) -> str:
    y, m, _ = str(period_end).split("-")
    return f"{y}Q{_QMAP[m]}"


def _prev_year_q(q: str) -> str:
    return f"{int(q[:4]) - 1}Q{q[-1]}"


def _yoy_by_quarter(df: pd.DataFrame, col: str) -> pd.Series:
    """按**日历季度**对齐计算同比。

    注意：不能用 shift(4)。本表缺 2018Q4（Datadog 2018 年无 10-K，Q4 无法由
    年度减前三季倒算），此时 shift(4) 会静默地把 2019Q4 与 2018Q3 相除，
    造出一个错误的同比（122.5% 而非 87.7%）。缺基期时应留空并显式暴露。
    """
    idx = dict(zip(df["quarter"], df[col]))
    out = []
    for q, v in zip(df["quarter"], df[col]):
        base = idx.get(_prev_year_q(q))
        out.append((v / base - 1) * 100 if (base and pd.notna(v) and base != 0) else float("nan"))
    return pd.Series(out, index=df.index)


def quarterly_revenue() -> pd.DataFrame:
    d = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json",
                 source="sec", note="DDOG companyfacts")
    save_raw("sec", "companyfacts", d, url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json",
             note="DDOG XBRL company facts")
    tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
    units = d["facts"]["us-gaap"][tag]["units"]["USD"]

    q, fy = {}, {}
    for u in units:
        if not u.get("start") or u.get("form") not in ("10-Q", "10-K"):
            continue
        s = date.fromisoformat(u["start"])
        e = date.fromisoformat(u["end"])
        days = (e - s).days
        if 80 <= days <= 100:                      # 单季
            prev = q.get(u["end"])
            if not prev or u.get("filed", "") >= prev.get("filed", ""):
                q[u["end"]] = u                      # 保留最新申报口径
        elif 360 <= days <= 370 and u["form"] == "10-K":  # 全年
            fy[u["end"]] = u

    rows = {e: {"period_end": e, "revenue": v["val"]} for e, v in q.items()}

    # Q4 倒算
    for fy_end, v in fy.items():
        if fy_end in rows:
            continue
        y = fy_end[:4]
        prev3 = [rows.get(f"{y}-03-31"), rows.get(f"{y}-06-30"), rows.get(f"{y}-09-30")]
        if all(prev3):
            rows[fy_end] = {"period_end": fy_end, "revenue": v["val"] - sum(p["revenue"] for p in prev3)}

    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["period_end"]))
    df["quarter"] = df["period_end"].map(_q_label)
    df["revenue_yoy_pct"] = _yoy_by_quarter(df, "revenue")
    return df


def deferred_revenue() -> pd.DataFrame:
    """递延收入（流动 + 非流动）与 RPO —— billings 与 RPO 都是作业点名的 KPI。"""
    d = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json",
                 source="sec", note="DDOG deferred revenue / RPO")
    out: dict[str, dict] = {}
    tags = [
        ("ContractWithCustomerLiabilityCurrent", "deferred_revenue_current"),
        ("ContractWithCustomerLiabilityNoncurrent", "deferred_revenue_noncurrent"),
        ("RevenueRemainingPerformanceObligation", "rpo"),
    ]
    for tag, col in tags:
        fact = d["facts"]["us-gaap"].get(tag)
        if not fact:
            print(f"    (标签不存在：{tag})")
            continue
        for u in fact["units"]["USD"]:
            if u.get("start"):          # 只要时点值（资产负债表口径），不要区间值
                continue
            out.setdefault(u["end"], {})[col] = u["val"]
    df = pd.DataFrame([{"period_end": k, **v} for k, v in sorted(out.items())])
    if {"deferred_revenue_current", "deferred_revenue_noncurrent"} <= set(df.columns):
        df["deferred_revenue_total"] = df["deferred_revenue_current"] + df["deferred_revenue_noncurrent"]
    return df


def _all_filings() -> list[dict]:
    """Return the complete SEC filing index, including older archive segments.

    SEC keeps only the latest 1,000 filings in ``filings.recent``.  For DDOG that
    list starts in 2023, while the IPO-era filings needed for KPI history live in
    the linked ``submissions-*.json`` archive files.
    """
    subs = get_json(f"https://data.sec.gov/submissions/CIK{CIK}.json", source="sec",
                    note="DDOG filing index")
    save_raw("sec", "submissions", subs, url=f"https://data.sec.gov/submissions/CIK{CIK}.json",
             note="DDOG current filing index")

    segments = [subs["filings"]["recent"]]
    for archive in subs["filings"].get("files", []):
        name = archive["name"]
        url = f"https://data.sec.gov/submissions/{name}"
        payload = get_json(url, source="sec", note=f"DDOG historic filing index {name}")
        save_raw("sec", name.removesuffix(".json"), payload, url=url,
                 note=f"DDOG historic filing index {name}")
        segments.append(payload)

    filings = []
    seen = set()
    for segment in segments:
        for row in zip(segment.get("form", []), segment.get("filingDate", []),
                       segment.get("accessionNumber", []), segment.get("reportDate", []),
                       segment.get("primaryDocument", []), segment.get("items", [])):
            form, filed, accession, report_date, document, items = row
            if accession in seen:
                continue
            seen.add(accession)
            filings.append({"form": form, "filing_date": filed, "accession": accession,
                            "report_date": report_date, "document": document, "items": items or ""})
    return filings


def _eight_k_kpis() -> pd.DataFrame:
    """从 8-K 新闻稿（EX-99.1）提取客户数与季度口径指标。

    关键点：**期间必须从 exhibit 文件名解析**（如 `ex-991x20260630x8k.htm` → 2026-06-30）。
    早前版本用正文正则 "quarter ended ..." 会匹配到**对比期**，导致客户数整体错位一个季度。
    """
    # Earnings releases are normally reported under Item 2.02.  Restricting by
    # item avoids downloading unrelated financing/personnel 8-Ks, while still
    # reaching the 2019 IPO-era archive segment.
    eights = [f for f in _all_filings()
              if f["form"] == "8-K" and f["filing_date"] >= "2019-01-01"
              and "2.02" in f["items"]]

    rows = []
    for filing in eights:
        fdate, accn, report_date = (filing["filing_date"],
                                    filing["accession"].replace("-", ""),
                                    filing["report_date"])
        try:
            idx = get_json(f"https://www.sec.gov/Archives/edgar/data/1561550/{accn}/index.json",
                           source="sec", note=f"8-K index {accn}")
        except Exception as e:
            print(f"    skip {fdate}: {str(e)[:60]}")
            continue
        names = [it["name"] for it in idx.get("directory", {}).get("item", [])]
        ex = [n for n in names if re.search(r"ex-?991", n, re.I) and n.lower().endswith((".htm", ".html"))] \
            or [n for n in names if re.search(r"ex-?99", n, re.I) and n.lower().endswith((".htm", ".html"))]
        if not ex:
            continue  # 非财报类 8-K（并购、债券等），正常跳过
        name = ex[0]

        # —— 期间：优先取 exhibit 文件名里编码的期末日 ——
        m = re.search(r"x(\d{8})x", name)
        period_end = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
                      if m else report_date)
        if not period_end:
            print(f"    skip {fdate}: 缺失报告期")
            continue

        url = f"{BASE}/{accn}/{name}"
        try:
            html = fetch(url, source="sec", note=f"DDOG 8-K EX-99.1 {fdate}").text
        except Exception as e:
            print(f"    skip {fdate}: {str(e)[:60]}")
            continue
        txt = re.sub(r"\s+", " ", re.sub(r"&nbsp;?", " ", re.sub(r"<[^>]+>", " ", html)))

        row = {"period_end": period_end, "filing_date": fdate,
               "customers_100k_source_url": url}

        # 标题交叉验证："Datadog Announces Second Quarter 2026 Financial Results"
        qm = re.search(r"Announces (First|Second|Third|Fourth) Quarter (\d{4}) Financial Results", txt)
        if qm:
            row["headline_quarter"] = f"{qm.group(2)}Q{['First','Second','Third','Fourth'].index(qm.group(1)) + 1}"

        m = re.search(r"([\d,]{3,})\s*customers with ARR of \$100,000 or more", txt, re.I)
        if m:
            row["customers_100k"] = int(m.group(1).replace(",", ""))

        m = re.search(
            r"(?:remaining performance obligations(?:,?\s*or\s*RPO)?|RPO)\s*"
            r"(?:was|stood at|of)\s*\$?\s*([\d.]+)\s*(million|billion)",
            txt, re.I,
        )
        if m:
            row["rpo_8k"] = float(m.group(1)) * (1e9 if m.group(2).lower() == "billion" else 1e6)

        m = re.search(r"revenue grew (\d+)% year-over-year to \$([\d.]+) (billion|million)", txt, re.I)
        if m:
            mult = 1e9 if m.group(3).lower() == "billion" else 1e6
            row["revenue_yoy_pct_8k"] = float(m.group(1))
            row["revenue_8k"] = float(m.group(2)) * mult

        # 季度自由现金流：用新闻稿要点句，避免误取表里的"年初至今"列
        m = (re.search(r"with free cash flow of \$([\d.]+) (million|billion)", txt, re.I)
             or re.search(r"and \$([\d.]+) (million|billion) in free cash flow", txt, re.I))
        if m:
            row["free_cash_flow"] = float(m.group(1)) * (1e9 if m.group(2).lower() == "billion" else 1e6)

        rows.append(row)
        print(f"    {period_end} (申报 {fdate}): 客户={row.get('customers_100k', '-'):<6} "
              f"收入={row.get('revenue_8k', '-')} FCF={row.get('free_cash_flow', '-')}")
    return pd.DataFrame(rows).sort_values("filing_date").drop_duplicates("period_end", keep="last") if rows else pd.DataFrame()


def _rpo_from_periodic_filings() -> pd.DataFrame:
    """从 10-Q / 10-K 正文回填早期 RPO。

    DDOG 在 2021 年部分申报中未将 RPO 写入 companyfacts 的标准 XBRL 标签，
    但数值已在财报正文披露；不能把 "XBRL 中没有" 误判成 "公司没有披露"。
    """
    rows = []
    for filing in _all_filings():
        form, end, acc, document = (filing["form"], filing["report_date"],
                                    filing["accession"], filing["document"])
        if form not in ("10-Q", "10-K") or end < "2019-01-01" or not document:
            continue
        url = f"{BASE}/{acc.replace('-', '')}/{document}"
        try:
            html = fetch(url, source="sec", note=f"DDOG {form} RPO {end}").text
        except Exception as exc:
            print(f"    RPO skip {end}: {str(exc)[:60]}")
            continue
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        match = re.search(
            r"aggregate transaction price allocated to remaining performance obligations[^.]{0,220}?"
            r"\$\s*([\d.]+)\s*(million|billion)",
            text, re.I,
        )
        if not match:
            continue
        value = float(match.group(1)) * (1e9 if match.group(2).lower() == "billion" else 1e6)
        rows.append({"period_end": end, "rpo_filing": value, "rpo_source_url": url})
    return pd.DataFrame(rows).drop_duplicates("period_end", keep="first") if rows else pd.DataFrame()


def main() -> None:
    print("== 目标变量采集开始（SEC EDGAR） ==")
    rev = quarterly_revenue()
    print(f"  季度收入：{len(rev)} 个季度，{rev['quarter'].iloc[0]} → {rev['quarter'].iloc[-1]}")

    dr = deferred_revenue()
    print(f"  递延收入：{len(dr)} 个时点")
    print("  8-K 新闻稿：")
    kpi = _eight_k_kpis()
    print("  10-Q / 10-K RPO 正文回填：")
    rpo_text = _rpo_from_periodic_filings()

    df = rev.merge(dr, on="period_end", how="left")
    if len(rpo_text):
        df = df.merge(rpo_text, on="period_end", how="left")
        df["rpo"] = df["rpo"].combine_first(df["rpo_filing"])
        df = df.drop(columns=["rpo_filing"])
    if len(kpi):
        df = df.merge(kpi, on="period_end", how="left")
        if "rpo_8k" in df:
            df["rpo"] = df["rpo"].combine_first(df["rpo_8k"])
        # 交叉校验：文件名解析出的期间应与标题里的季度一致
        bad = df[df["headline_quarter"].notna() & (df["headline_quarter"] != df["quarter"])]
        if len(bad):
            print(f"  ! 期间校验不一致 {len(bad)} 行：\n{bad[['quarter', 'headline_quarter', 'period_end']]}")

    if "deferred_revenue_total" in df.columns:
        df["billings"] = df["revenue"] + df["deferred_revenue_total"].diff()
    elif "deferred_revenue_current" in df.columns:
        df["billings"] = df["revenue"] + df["deferred_revenue_current"].diff()
    if "customers_100k" in df.columns:
        df["customers_100k_yoy_pct"] = _yoy_by_quarter(df, "customers_100k")

    df["source"] = "sec"
    df["legal_basis"] = LEGAL_BASIS["sec"]
    cols = [c for c in ["quarter", "period_end", "headline_quarter", "revenue", "revenue_yoy_pct",
                        "revenue_yoy_pct_8k", "billings", "rpo",
                        "deferred_revenue_current", "deferred_revenue_noncurrent",
                        "deferred_revenue_total", "customers_100k", "customers_100k_yoy_pct",
                        "free_cash_flow", "filing_date", "customers_100k_source_url", "rpo_source_url",
                        "source", "legal_basis"]
            if c in df.columns]
    df = df[cols]

    print("\n== 写出 ==")
    for p in write_table(df, "ddog_target_quarterly"):
        print("  ", p)
    print(f"\n完成：{len(df)} 个季度")
    print(df.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
