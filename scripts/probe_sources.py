#!/usr/bin/env python3
"""Probe candidate high-frequency / alternative data sources for DDOG research.

Runs read-only HTTP GETs against public APIs and reports, per source:
  status, latency, payload size, a small sample, and how long a history it covers.

⚠️ 合规修正（2026-09-14）
    本脚本的早期版本直接用裸 requests 发请求，在探索阶段**曾成功抓取
    `docs.datadoghq.com/integrations/` 与 `status.datadoghq.com/history.rss`**。
    这违反了 Datadog Acceptable Use Policy 的 "No Framing or Scraping" 条款
    （见 docs/data_source_strategy.md 第 3 节）——而本作业正是 Datadog 出的。

    现已整改：全部探测改为经 scripts/fetch_utils.py 的合规出口（域名黑名单 + 限速 + UA 声明），
    `*.datadoghq.com` 已被硬编码拦截；历史探索记录在 probe_results.json 中标注为
    `compliance_violation_historical`，并已从数据管道中移除对应来源（原先计划的
    S4「定价页/集成目录」已改用开源仓库与官方 RSS 替代）。

Usage:  .venv/bin/python scripts/probe_sources.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import ComplianceError, NotFound, fetch  # noqa: E402

results = []


def probe(name, url, note="", sample=None, headers=None, parse=None):
    """所有探测必须走合规出口；被黑名单拦截的源记为 BLOCKED_BY_POLICY。"""
    t0 = time.time()
    try:
        r = fetch(url, source=name.split()[0].lower(), note=note, tries=1, timeout=45)
        ms = int((time.time() - t0) * 1000)
        info = ""
        if parse and r.ok:
            try:
                info = parse(r)
            except Exception as e:  # noqa: BLE001
                info = f"parse-error: {e}"
        results.append({"source": name, "status": r.status_code, "ms": ms,
                        "bytes": len(r.content), "detail": info, "note": note, "url": url})
        print(f"[{r.status_code}] {name:<42} {len(r.content):>9,}B {ms:>5}ms  {info}")
    except ComplianceError:
        results.append({"source": name, "status": "BLOCKED_BY_POLICY",
                        "detail": "blocked by local compliance blacklist (Datadog AUP)",
                        "note": note, "url": url})
        print(f"[BLOCKED] {name:<38} 被本地合规黑名单拦截（Datadog AUP 禁止抓取其站点）")
    except NotFound:
        results.append({"source": name, "status": 404, "detail": "not found", "note": note, "url": url})
        print(f"[404] {name:<42} 目标不存在")
    except Exception as e:  # noqa: BLE001
        results.append({"source": name, "status": "ERR", "detail": str(e)[:120], "note": note, "url": url})
        print(f"[ERR] {name:<42} {e}")


# ---------------------------------------------------------------- 1. target KPI
def sec_parse(r):
    d = r.json()
    rev = d["facts"]["us-gaap"].get("RevenueFromContractWithCustomerExcludingAssessedTax", {})
    q = [u for u in rev.get("units", {}).get("USD", []) if u.get("form") in ("10-Q", "10-K") and u.get("start")]
    q = sorted({(u["start"], u["end"], u["val"]) for u in q})
    if not q:
        return "no quarterly points"
    return f"{len(q)} revenue datapoints, latest end={q[-1][1]} val=${q[-1][2]:,}"


probe("SEC XBRL companyfacts (revenue)", "https://data.sec.gov/api/xbrl/companyfacts/CIK0001561550.json", parse=sec_parse)
probe(
    "SEC XBRL companyconcept (Revenues)",
    "https://data.sec.gov/api/xbrl/companyconcept/CIK0001561550/us-gaap/Revenues.json",
    note="authoritative quarterly target variable",
)
probe(
    "SEC full-text search (8-K/10-Q)",
    "https://efts.sec.gov/LATEST/search-index?q=%22Datadog%22&forms=10-Q",
    note="filing discovery",
)

# ------------------------------------------------- 2. developer/instrumentation
probe(
    "npm dd-trace download range",
    "https://api.npmjs.org/downloads/range/2024-01-01:2026-09-01/dd-trace",
    parse=lambda r: (
        lambda d: f"{len(d['downloads'])} daily points, total={sum(x['downloads'] for x in d['downloads']):,}"
    )(r.json()),
)
probe(
    "npm datadog-lambda-js",
    "https://api.npmjs.org/downloads/range/2024-01-01:2026-09-01/datadog-lambda-js",
    parse=lambda r: f"{sum(x['downloads'] for x in r.json()['downloads']):,} downloads in window",
)
probe(
    "PyPI ddtrace (pypistats daily)",
    "https://pypistats.org/api/packages/ddtrace/overall",
    parse=lambda r: f"{len(r.json()['data'])} daily points, latest {r.json()['data'][-1]}",
)
probe(
    "Terraform Registry DataDog provider",
    "https://registry.terraform.io/v1/providers/DataDog/datadog",
    parse=lambda r: f"version={r.json().get('version')} downloads={r.json().get('downloads'):,}",
)
probe(
    "Terraform Registry provider downloads endpoint",
    "https://registry.terraform.io/v1/providers/DataDog/datadog/downloads",
    parse=lambda r: f"{len(r.json())} version buckets",
)
probe(
    "Docker Hub datadog/agent pulls",
    "https://hub.docker.com/v2/repositories/datadog/agent/",
    parse=lambda r: f"pulls={r.json().get('pull_count'):,} stars={r.json().get('star_count')}",
)
probe(
    "GitHub datadog-agent release assets",
    "https://api.github.com/repos/DataDog/datadog-agent/releases?per_page=5",
    parse=lambda r: "; ".join(
        f"{x['tag_name']}:{sum(a['download_count'] for a in x['assets']):,}dl" for x in r.json()
    ),
)
probe(
    "ArtifactHub datadog helm chart",
    "https://artifacthub.io/api/v1/packages/helm/datadog/datadog",
    parse=lambda r: f"version={r.json().get('version')} ts={r.json().get('ts')}",
)
probe(
    "[黑名单自测] Datadog docs 集成目录",
    "https://docs.datadoghq.com/integrations/",
    note="⚠️ 不得抓取：Datadog AUP 'No Framing or Scraping'。此处仅用于验证本地黑名单生效，"
         "预期结果为 BLOCKED_BY_POLICY。产品面数据已改用 GitHub 开源仓库替代。",
)

# ------------------------------------------------------- 3. demand-side / hiring
probe(
    "Greenhouse board API (Datadog jobs)",
    "https://boards-api.greenhouse.io/v1/boards/datadog/jobs",
    parse=lambda r: (
        lambda d: f"{len(d['jobs'])} open roles; latest updated={max(j['updated_at'] for j in d['jobs'])[:10]}"
    )(r.json()),
)
probe(
    "Greenhouse departments (Datadog)",
    "https://boards-api.greenhouse.io/v1/boards/datadog/departments",
    parse=lambda r: ", ".join(f"{d['name']}({len(d['jobs'])})" for d in r.json()["departments"])[:180],
)
probe(
    "DOL OFLC LCA disclosure (H-1B, quarterly xlsx)",
    "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q2.xlsx",
    note="employer-level hiring incl. Datadog, free & legal",
)
probe(
    "DOL OFLC performance data landing",
    "https://www.dol.gov/agencies/eta/foreign-labor/performance",
)

# --------------------------------------------------------- 4. web footprint
probe(
    "crt.sh certificate transparency",
    "https://crt.sh/?q=%25.datadoghq.com&output=json&exclude=expired",
    parse=lambda r: (
        lambda d: f"{len(d)} certs, {len({n for c in d for n in c['name_value'].split(chr(10))})} distinct names"
    )(r.json()),
)
probe(
    "Wayback CDX pricing-page history",
    "http://web.archive.org/cdx/search/cdx?url=datadoghq.com/pricing&output=json",
    parse=lambda r: f"{max(0, len(r.json()) - 1)} snapshots",
)
probe(
    "Wikipedia pageviews (Datadog, monthly)",
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
    "Datadog/monthly/2023010100/2026090100",
    parse=lambda r: f"{len(r.json()['items'])} months, last={r.json()['items'][-1]['views']:,} views",
)
probe(
    "Hacker News Algolia mentions",
    "https://hn.algolia.com/api/v1/search?query=datadog&numericFilters=created_at_i%3E1704067200&hitsPerPage=1",
    parse=lambda r: f"{r.json()['nbHits']:,} hits since 2024-01-01",
)
probe(
    "Common Crawl columnar index",
    "http://index.commoncrawl.org/CC-MAIN-2025-13-index?url=datadoghq.com&output=json&limit=3",
    note="monthly web-wide crawl for tech adoption",
)

# ------------------------------------------------------------ 5. market / macro
probe(
    "Nasdaq historical prices (DDOG)",
    "https://api.nasdaq.com/api/quote/DDOG/historical?assetclass=stocks&fromdate=2024-01-01&limit=10",
    headers={"User-Agent": "Mozilla/5.0"},
)
probe("FRED CSV (10y treasury)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
probe(
    "[黑名单自测] Datadog 状态页 RSS",
    "https://status.datadoghq.com/history.rss",
    note="⚠️ 不得抓取：即使是官方 RSS，AUP 也禁止自动装置从 Site 收集内容；"
         "本作业由 Datadog 出题，故采取最保守立场。预期结果为 BLOCKED_BY_POLICY。",
)

print("\n=== summary ===")
ok = [r for r in results if r["status"] == 200]
print(f"{len(ok)}/{len(results)} sources reachable")
for r in results:
    if r["status"] != 200:
        print(f"  BLOCKED/ERR {r['source']}: {r['status']} {r.get('detail','')[:80]}")

with open("probe_results.json", "w") as f:
    json.dump({"ran_at": datetime.now(timezone.utc).isoformat(), "results": results}, f, indent=2)
print("wrote probe_results.json")
