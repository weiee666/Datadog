"""S2 历史补充 · Datadog 的长期招聘/人力轨迹（免费公开数据）。

S2 的日频信号（Greenhouse）只能从今天起逐日累积，无法回溯。本脚本用两个**免费、合法、
可回溯多年**的公开数据源把历史补齐：

  ① 美国劳工部 OFLC 的 H-1B/LCA 披露数据（季度 xlsx，政府公开记录）
     - 优点：可回溯到 2019Q4（FY2020Q1），含职位名称、SOC 代码、薪资、工作地、
       "新雇 vs 续用"标志，能直接构造「季度新雇强度」序列。
     - 用、EMPLOYER_FEIN 精确匹配公司实体，避免"名字里含 Datadog"的误伤。
     - 局限：只覆盖 H-1B 赞助岗位（约占总招聘的一部分，且偏向工程岗），
       且 LCA 是"申请"而非"实际入职"（申请需在岗位开始前提交，因此略领先）。
  ② SEC 10-K 的员工总数（年度锚点）
     - DDOG 在 10-K 里披露 "we had X employees"，可做长期人力水平的锚。

产出：
  data/raw/dol/<FY>_<Q>_datadog.csv        过滤后的 Datadog 明细（含全部原始列）
  data/raw/dol/<FY>_<Q>_meta.json          来源 URL / 文件 sha256 / 扫描行数（可审计）
  data/processed/s2_dol_lca_quarterly.csv  季度面板（宽表）
  data/processed/s2_dol_lca_long.csv       长表，可直接并入 observations
  data/processed/s2_headcount_annual.csv   10-K 年度员工数

用法：
  .venv/bin/python scripts/collect_s2_history.py              # 全部季度
  .venv/bin/python scripts/collect_s2_history.py --years 2024 2025 2026
  .venv/bin/python scripts/collect_s2_history.py --headcount-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import (LEGAL_BASIS, RAW_DIR, SESSION, MIN_INTERVAL,  # noqa: E402
                         get_json, save_raw, write_table)

DOL = "https://www.dol.gov"
FLAT = DOL + "/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY{fy}_Q{q}.xlsx"
NEW = DOL + "/media/LCA_Disclosure_Data_FY{fy}_Q{q}.xlsx"   # FY2026+ 改到 /media/
DOL_DIR = RAW_DIR / "dol"

EMPLOYER_RE = re.compile(r"DATADOG", re.I)
ENG_RE = re.compile(
    r"software|engineer|developer|architect|sre|site reliability|data scien|"
    r"machine learning|infrastructure|platform|security|backend|frontend|full stack",
    re.I,
)
_WAGE_MULT = {"YEAR": 1.0, "HOUR": 2080.0, "WEEK": 52.0, "BI-WEEKLY": 26.0, "MONTH": 12.0}

# 固定 schema：DOL 各年份文件的列结构会变（FY2020 没有 EMPLOYER_FEIN 列），
# 统一裁剪到这套列，缺失的补 None，下游代码就不必到处判空。
WANTED = [
    "CASE_NUMBER", "CASE_STATUS", "RECEIVED_DATE", "DECISION_DATE", "VISA_CLASS",
    "JOB_TITLE", "SOC_CODE", "SOC_TITLE", "FULL_TIME_POSITION", "BEGIN_DATE",
    "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT", "CHANGE_EMPLOYER",
    "TOTAL_WORKER_POSITIONS", "EMPLOYER_NAME", "EMPLOYER_FEIN",
    "WORKSITE_CITY", "WORKSITE_STATE", "WAGE_RATE_OF_PAY_FROM", "WAGE_UNIT_OF_PAY",
    "PREVAILING_WAGE",
]


def quarter_range(fy_from: int = 2020, fy_to: int = 2026) -> list[tuple[int, int, list[str]]]:
    """生成 (FY, Q, [候选URL...])。DOL 财年：Q1=10-12月, Q2=1-3月, Q3=4-6月, Q4=7-9月。

    ⚠️ DOL 的存放目录在不同年份间变动过：FY2025 及以前平铺在 pdfs/ 下，
    FY2026 只有 Q3 在 /media/，而 FY2026 Q1 仍在平铺目录。因此每个季度都试两个路径。
    """
    out = []
    for fy in range(fy_from, fy_to + 1):
        for q in (1, 2, 3, 4):
            out.append((fy, q, [FLAT.format(fy=fy, q=q), NEW.format(fy=fy, q=q)]))
    return out


def cal_quarter(fy: int, q: int) -> str:
    """DOL 财年季度 -> 自然季度标签，便于与 DDOG 财报对齐。"""
    return f"{fy - 1}Q4" if q == 1 else f"{fy}Q{q - 1}"


_QEND = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def q_end(label: str) -> str:
    """季度标签 -> 该季度最后一天（'2024Q1' -> '2024-03-31'）。

    observations 规范表要求 date 列是真正的日期，因此季度序列统一锚在季末。
    """
    return f"{int(label[:4])}-{_QEND[int(label[-1])]}"


def download(url: str, dest: Path, tries: int = 3) -> tuple[int, str]:
    """流式下载大文件（50–150MB），返回 (字节数, sha256)。"""
    last = None
    for attempt in range(1, tries + 1):
        try:
            with SESSION.get(url, stream=True, timeout=180) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                h = hashlib.sha256()
                n = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
                return n, h.hexdigest()
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败 {url}: {last}")


def extract_datadog(path: Path) -> tuple[pd.DataFrame, int, int]:
    """流式解析 xlsx，只保留雇主名含 DATADOG 的行。返回 (明细, 总行数, 表头列数)。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    hdr = [str(h).strip().upper() if h is not None else "" for h in next(it)]
    idx = {h: i for i, h in enumerate(hdr) if h}
    name_i, dba_i = idx.get("EMPLOYER_NAME"), idx.get("TRADE_NAME_DBA")

    rows, total = [], 0
    for row in it:
        total += 1
        name = str(row[name_i]) if name_i is not None and row[name_i] else ""
        dba = str(row[dba_i]) if dba_i is not None and row[dba_i] else ""
        if not (EMPLOYER_RE.search(name) or EMPLOYER_RE.search(dba)):
            continue
        rows.append({c: (row[idx[c]] if c in idx else None) for c in WANTED})
    wb.close()
    return pd.DataFrame(rows, columns=WANTED), total, len(hdr)


def collect_lca(years: list[int] | None = None) -> pd.DataFrame:
    DOL_DIR.mkdir(parents=True, exist_ok=True)
    targets = quarter_range()
    if years:
        targets = [t for t in targets if t[0] in years]

    frames = []
    for fy, q, urls in targets:
        tag = f"FY{fy}_Q{q}"
        det_path = DOL_DIR / f"{tag}_datadog.csv"
        meta_path = DOL_DIR / f"{tag}_meta.json"
        if det_path.exists() and meta_path.exists():
            print(f"  {tag}: 已有缓存，跳过")
            frames.append(pd.read_csv(det_path).reindex(columns=WANTED).assign(_fy=fy, _q=q))
            continue

        t0 = time.time()
        got = None
        for url in urls:
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "lca.xlsx"
                    size, digest = download(url, tmp)
                    df, total, ncol = extract_datadog(tmp)
                got = (url, size, digest, df, total, ncol)
                break
            except FileNotFoundError:
                continue                      # 换下一个候选路径
            except Exception as e:  # noqa: BLE001
                print(f"  {tag}: {url.rsplit('/', 1)[-1]} 失败 {str(e)[:60]}")
                continue
        if got is None:
            print(f"  {tag}: 文件不存在（该季度尚未发布），跳过")
            continue
        url, size, digest, df, total, ncol = got

        if not len(df):
            print(f"  {tag}: 扫描 {total:,} 行，未命中 Datadog")
        else:
            fein = df["EMPLOYER_FEIN"].dropna().astype(str)
            fein = fein[fein.str.strip() != ""].value_counts()
            fein_txt = f" FEIN={dict(fein.head(3))}" if len(fein) else "（该年份文件无 FEIN 列）"
            print(f"  {tag}: 扫描 {total:,} 行({ncol}列) -> Datadog {len(df)} 条 "
                  f"[{time.time() - t0:.0f}s, {size / 1e6:.0f}MB]{fein_txt}")
        df.to_csv(det_path, index=False, encoding="utf-8-sig")
        meta_path.write_text(json.dumps({
            "source_url": url, "bytes": size, "sha256": digest, "rows_scanned": total,
            "columns": ncol, "datadog_rows": len(df),
            "legal_basis": LEGAL_BASIS["dol"], "fetched_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        frames.append(df.assign(_fy=fy, _q=q))
        time.sleep(MIN_INTERVAL)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def annualize_wage(v, unit) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    m = _WAGE_MULT.get(str(unit).strip().upper(), None)
    return x * m if m else None


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame()
    df = df.copy()
    df["quarter"] = [cal_quarter(int(a), int(b)) for a, b in zip(df["_fy"], df["_q"])]
    df["is_eng"] = df["JOB_TITLE"].fillna("").str.contains(ENG_RE)
    df["new_emp"] = pd.to_numeric(df.get("NEW_EMPLOYMENT"), errors="coerce").fillna(0)
    df["wage_annual"] = [annualize_wage(v, u) for v, u in
                         zip(df.get("WAGE_RATE_OF_PAY_FROM", []), df.get("WAGE_UNIT_OF_PAY", []))]

    g = df.groupby("quarter")
    out = pd.DataFrame({
        "lca_cases": g.size(),
        "lca_certified": g["CASE_STATUS"].apply(lambda s: (s.astype(str).str.lower() == "certified").sum()),
        "lca_new_employment": g["new_emp"].sum(),
        "lca_engineering_cases": g["is_eng"].sum(),
        "lca_median_wage_annual": g["wage_annual"].median().round(0),
        "lca_worksite_states": g["WORKSITE_STATE"].nunique(),
    }).reset_index()
    out["lca_new_employment_share"] = (out["lca_new_employment"] / out["lca_cases"].replace(0, pd.NA)) * 100
    return out.sort_values("quarter").reset_index(drop=True)


def _all_filings(form: str) -> list[tuple[str, str]]:
    """取某表单的全部历史申报（submissions 的 recent 只覆盖最近约 1000 份，需翻页）。"""
    subs = get_json("https://data.sec.gov/submissions/CIK0001561550.json", source="sec",
                    note="DDOG filing index (headcount)")
    out = [(d, a) for f, d, a in zip(subs["filings"]["recent"]["form"],
                                     subs["filings"]["recent"]["filingDate"],
                                     subs["filings"]["recent"]["accessionNumber"]) if f == form]
    for extra in subs["filings"].get("files", []):
        try:
            more = get_json(f"https://data.sec.gov/submissions/{extra['name']}", source="sec",
                            note=f"DDOG filings page {extra['name']}")
            out += [(d, a) for f, d, a in zip(more["form"], more["filingDate"],
                                              more["accessionNumber"]) if f == form]
        except Exception as e:  # noqa: BLE001
            print(f"  (翻页 {extra.get('name')} 失败: {str(e)[:50]})")
    return out


def _clean_html(html: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), t)   # 数字实体，如 &#160;
    t = t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8217;", "'")
    return re.sub(r"\s+", " ", t)


def collect_headcount() -> pd.DataFrame:
    """从 10-K 提取员工总数 / 研发人数 / 覆盖国家数（年度锚点）。

    10-K 原文措辞（实测）：
      "As of December 31, 2025, we had approximately 8,100 employees operating across 35 countries."
      "As of December 31, 2025, we had approximately 3,900 employees in our research and development organization."
    """
    from fetch_utils import fetch

    rows = []
    for fdate, acc in sorted(_all_filings("10-K")):
        accn = acc.replace("-", "")
        try:
            idx = get_json(f"https://www.sec.gov/Archives/edgar/data/1561550/{accn}/index.json",
                           source="sec", note=f"10-K index {accn}")
            names = [i["name"] for i in idx["directory"]["item"]]
            main = [n for n in names if re.match(r"^ddog-.*\.htm$", n, re.I)] \
                or [n for n in names if n.endswith(".htm") and not n.startswith("R")]
            if not main:
                continue
            # 报表期末从文件名取（ddog-20251231.htm → 2025-12-31）。注意 10-K 在次年 2 月申报，
            # 用申报年份当财年会整体偏一年。
            pm = re.search(r"ddog-(\d{8})\.htm", main[0], re.I)
            fy = pm.group(1)[:4] if pm else str(int(fdate[:4]) - 1)
            period_end = (f"{pm.group(1)[:4]}-{pm.group(1)[4:6]}-{pm.group(1)[6:]}"
                          if pm else f"{fy}-12-31")
            txt = _clean_html(fetch(f"https://www.sec.gov/Archives/edgar/data/1561550/{accn}/{main[0]}",
                                    source="sec", note=f"DDOG 10-K {fdate}").text)
            # 注意：10-K 里先出现的是**各职能人数**（"we had approximately 3,900 employees in our
            # research and development organization"），总数在 Human Capital Management 一节。
            # 只取第一个匹配会把研发人数误当总数，故：
            #   ① 优先匹配带 "operating across N countries" 的总数句；
            #   ② 否则取所有匹配中的**最大值**（总数必然最大）。
            allc = [int(x.replace(",", ""))
                    for x in re.findall(r"we had (?:approximately )?([\d,]{3,}) employees", txt, re.I)]
            m = re.search(r"we had (?:approximately )?([\d,]{3,}) employees"
                          r" operating across ([\d,]{1,3}) countries", txt, re.I)
            if not m and not allc:
                print(f"  10-K {fdate}: 未匹配到员工数")
                continue
            row = {"filing_date": fdate, "fiscal_year": fy, "period_end": period_end,
                   "employees": int(m.group(1).replace(",", "")) if m else max(allc),
                   "form": "10-K"}
            if m and m.group(2):
                row["countries"] = int(m.group(2).replace(",", ""))
            m2 = re.search(r"we had (?:approximately )?([\d,]{3,}) employees in our "
                           r"research and development", txt, re.I)
            if m2:
                row["employees_rnd"] = int(m2.group(1).replace(",", ""))
            rows.append(row)
            print(f"  10-K {fdate} (FY{row['fiscal_year']}): {row['employees']:,} 人"
                  + (f"，研发 {row['employees_rnd']:,}" if "employees_rnd" in row else "")
                  + (f"，{row['countries']} 国" if "countries" in row else ""))
        except Exception as e:  # noqa: BLE001
            print(f"  10-K {fdate}: 失败 {str(e)[:60]}")

    df = pd.DataFrame(rows).drop_duplicates("fiscal_year", keep="last").sort_values("fiscal_year")
    if len(df):
        df["employees_yoy_pct"] = df["employees"].pct_change() * 100
        df["source"] = "sec"
        df["legal_basis"] = LEGAL_BASIS["sec"]
    return df.reset_index(drop=True) if len(df) else df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, help="只抓指定财年，如 --years 2024 2025")
    ap.add_argument("--headcount-only", action="store_true")
    a = ap.parse_args()

    if a.headcount_only:
        print("== 10-K 员工数 ==")
        hc = collect_headcount()
        if len(hc):
            for p in write_table(hc, "s2_headcount_annual"):
                print("  ", p)
        return

    print("== S2 历史补充：DOL H-1B/LCA 季度披露数据 ==")
    print(f"   合规依据：{LEGAL_BASIS['dol']}")
    raw = collect_lca(a.years)
    if not len(raw):
        print("没有取到数据。")
        return

    panel = aggregate(raw)
    print("\n== 季度面板 ==")
    print(panel.to_string(index=False))

    print("\n== 写出 ==")
    for p in write_table(panel, "s2_dol_lca_quarterly"):
        print("  ", p)

    # 长表：直接并入 observations
    spec = {"lca_cases": ("cases", "LCA(H-1B) 申请件数"),
            "lca_certified": ("cases", "其中获批件数"),
            "lca_new_employment": ("positions", "新雇岗位数（区别于续用）"),
            "lca_engineering_cases": ("cases", "工程类岗位申请件数"),
            "lca_median_wage_annual": ("USD/year", "薪资中位数（年化）"),
            "lca_worksite_states": ("states", "覆盖州数")}
    recs = []
    for _, r in panel.iterrows():
        for col, (unit, _desc) in spec.items():
            v = r[col]
            if pd.isna(v):
                continue
            recs.append({"date": q_end(r["quarter"]), "series_id": f"s2.dol.{col}",
                         "value": float(v), "unit": unit, "freq": "Q", "source": "dol",
                         "legal_basis": LEGAL_BASIS["dol"]})
    long = pd.DataFrame(recs)
    for p in write_table(long, "s2_dol_lca_long"):
        print("  ", p)

    print("\n== 10-K 员工数（年度锚点） ==")
    hc = collect_headcount()
    if len(hc):
        for p in write_table(hc, "s2_headcount_annual"):
            print("  ", p)

    print(f"\n完成：{len(panel)} 个季度（{panel['quarter'].iloc[0]} → {panel['quarter'].iloc[-1]}），"
          f"明细 {len(raw):,} 条")


if __name__ == "__main__":
    main()
