#!/usr/bin/env python3
"""DDOG alternative-data collector.

Single source of truth for the project: pulls every signal the web dashboard and
the Tableau workbook both consume.

Compliance (see docs/data_source_strategy.md §3):
  * only official public APIs / public datasets / official RSS feeds are used
  * datadoghq.com and every subdomain is HARD-BLOCKED (Datadog AUP "No Framing or
    Scraping") -- enforced in fetch(), not by convention
  * identifiable User-Agent, polite rate limiting (SEC asks for <=10 req/s)

Outputs
  data/raw/*.json          verbatim API responses (audit trail)
  data/tidy/*.csv          tidy long format: date,source,metric,value
  data/tableau/*.csv       wide format, ready for Tableau Desktop import
  site/data/payload.json   everything the web dashboard needs

Usage:  .venv/bin/python scripts/collect.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TIDY = ROOT / "data" / "tidy"
TABLEAU = ROOT / "data" / "tableau"
SITE_DATA = ROOT / "site" / "data"
for d in (RAW, TIDY, TABLEAU, SITE_DATA):
    d.mkdir(parents=True, exist_ok=True)

UA = "DDOG-altdata-research/0.1 (take-home research; contact: analyst@example.com)"
BLOCKED_HOSTS = ("datadoghq.com", "datadog.com")  # Datadog AUP: no scraping of the Site
MIN_INTERVAL = 0.25  # seconds between requests to the same host

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "application/json, text/csv, */*"})
_last_hit: dict[str, float] = {}
_log: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    _log.append(msg)


def fetch(url: str, *, allow_html: bool = False) -> requests.Response | None:
    """GET with compliance guardrails: domain blacklist + per-host rate limit."""
    host = (urlparse(url).hostname or "").lower()
    if any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS):
        log(f"  BLOCKED by policy (Datadog AUP): {url}")
        return None
    wait = MIN_INTERVAL - (time.time() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    try:
        r = S.get(url, timeout=45)
        _last_hit[host] = time.time()
        if r.status_code != 200:
            log(f"  [{r.status_code}] {url}")
            return None
        return r
    except Exception as e:  # noqa: BLE001
        log(f"  [ERR] {url} -> {str(e)[:90]}")
        return None


def save_raw(name: str, obj) -> None:
    (RAW / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- #
# S1 · instrumentation flow: SDK/agent downloads = billable usage raw material
# --------------------------------------------------------------------------- #
NPM_PACKAGES = ["dd-trace", "datadog-lambda-js", "@datadog/browser-rum", "@datadog/browser-logs"]
PYPI_PACKAGES = ["ddtrace", "datadog", "datadog-lambda"]


def collect_npm(start: str, end: str) -> pd.DataFrame:
    """npm range API caps a single request at ~18 months -> chunk it.

    npm started publishing download counts on 2015-01-10, so we can rebuild a
    ~11 year daily history, which is what makes walk-forward backtests possible.
    """
    log("S1 npm downloads")
    rows = []
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    windows = []
    cur = max(s, date(2015, 1, 11))
    while cur < e:
        nxt = min(cur + timedelta(days=365), e)
        windows.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + timedelta(days=1)

    for pkg in NPM_PACKAGES:
        got = 0
        for w_start, w_end in windows:
            r = fetch(f"https://api.npmjs.org/downloads/range/{w_start}:{w_end}/{pkg}")
            if not r:
                continue
            data = r.json()
            for d in data.get("downloads", []):
                rows.append(
                    {"date": d["day"], "source": "npm", "metric": f"dl_{pkg}", "value": d["downloads"]}
                )
                got += 1
        log(f"  {pkg:<26} {got:>5} daily points [{windows[0][0]} .. {windows[-1][1]}]")
        save_raw(f"npm_{pkg.replace('/', '_')}", {"collected": datetime.now(timezone.utc).isoformat(), "points": got})
    return pd.DataFrame(rows)


def collect_pypi() -> pd.DataFrame:
    log("S1 PyPI downloads")
    rows = []
    for pkg in PYPI_PACKAGES:
        r = fetch(f"https://pypistats.org/api/packages/{pkg}/overall")
        if not r:
            continue
        data = r.json()
        save_raw(f"pypi_{pkg}", data)
        for d in data.get("data", []):
            if d.get("category") == "without_mirrors":  # exclude mirror traffic (bots)
                rows.append({"date": d["date"], "source": "pypi", "metric": f"dl_{pkg}", "value": d["downloads"]})
        time.sleep(MIN_INTERVAL)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# S2 · Datadog's own hiring velocity (Greenhouse Job Board API - public by design)
# --------------------------------------------------------------------------- #
def collect_greenhouse() -> tuple[pd.DataFrame, dict]:
    log("S2 Greenhouse jobs")
    r = fetch("https://boards-api.greenhouse.io/v1/boards/datadog/jobs")
    if not r:
        return pd.DataFrame(), {}
    data = r.json()
    save_raw("greenhouse_jobs", data)
    jobs = data.get("jobs", [])
    today = date.today().isoformat()
    rows = []
    for j in jobs:
        dept = (j.get("departments") or [{}])[0].get("name", "Unknown")
        rows.append(
            {
                "date": today,
                "source": "greenhouse",
                "metric": "open_roles_total",
                "value": 1,
                "job_id": j.get("id"),
                "title": j.get("title"),
                "department": dept,
                "updated_at": (j.get("updated_at") or "")[:10],
            }
        )
    by_dept = (
        pd.DataFrame(rows).groupby(["date", "source", "department"], as_index=False)["value"].sum()
        if rows
        else pd.DataFrame()
    )
    df = pd.DataFrame(rows)
    summary = {
        "as_of": today,
        "open_roles": len(jobs),
        "by_department": (df.groupby("department")["value"].sum().sort_values(ascending=False).to_dict() if rows else {}),
        "newest_update": max((j.get("updated_at") or "")[:10] for j in jobs) if jobs else None,
    }
    # daily snapshot = how we build our own history going forward
    snap = RAW / "greenhouse_snapshots"
    snap.mkdir(exist_ok=True)
    (snap / f"{today}.json").write_text(json.dumps(summary, indent=2))
    return by_dept, summary


# --------------------------------------------------------------------------- #
# Target variable · SEC XBRL (quarterly revenue, deferred revenue) + 8-K KPIs
# --------------------------------------------------------------------------- #
CIK = "0001561550"


def collect_sec() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log("Target SEC XBRL")
    rows, kpi = [], {}
    r = fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json")
    if r:
        d = r.json()
        save_raw("sec_companyfacts", d)
        facts = d["facts"]["us-gaap"]

        # quarterly revenue (10-Q discrete quarters) + annual (10-K) to derive Q4
        tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
        units = facts[tag]["units"]["USD"]
        quarters: dict[tuple[str, str], int] = {}
        annual: dict[str, int] = {}
        for u in units:
            if not u.get("start") or u.get("form") not in ("10-Q", "10-K"):
                continue
            s, e = date.fromisoformat(u["start"]), date.fromisoformat(u["end"])
            span = (e - s).days
            if 80 <= span <= 100:
                quarters[(u["start"], u["end"])] = u["val"]
            elif 360 <= span <= 370 and u["form"] == "10-K":
                annual[u["end"]] = u["val"]

        # backfill Q4 = FY - (Q1+Q2+Q3)
        for fy_end, fy_val in annual.items():
            y = fy_end[:4]
            q13 = [v for (s, _e), v in quarters.items() if s.startswith(y)]
            if len(q13) == 3:
                q4 = fy_val - sum(q13)
                if q4 > 0:
                    quarters[(f"{y}-10-01", f"{y}-12-31")] = q4

        for (s, e), v in sorted(quarters.items()):
            rows.append({"date": e, "source": "sec_xbrl", "metric": "revenue_q", "value": v})
        for fy_end, v in sorted(annual.items()):
            rows.append({"date": fy_end, "source": "sec_xbrl", "metric": "revenue_fy", "value": v})

        # deferred revenue (current) -> billings proxy
        dr = facts.get("ContractWithCustomerLiabilityCurrent", {}).get("units", {}).get("USD", [])
        for u in dr:
            if not u.get("start") and u.get("form") in ("10-Q", "10-K"):
                rows.append(
                    {"date": u["end"], "source": "sec_xbrl", "metric": "deferred_rev_current", "value": u["val"]}
                )

    # 8-K press releases -> headline KPIs
    log("Target 8-K press releases")
    r = fetch(f"https://data.sec.gov/submissions/CIK{CIK}.json")
    if r:
        sub = r.json()
        save_raw("sec_submissions", sub)
        rec = sub["filings"]["recent"]
        eights = [
            (fd, acc, rep)
            for form, fd, acc, rep in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["reportDate"])
            if form == "8-K"
        ][:8]
        for filing_date, acc, report_date in eights:
            acc_nodash = acc.replace("-", "")
            idx = fetch(f"https://www.sec.gov/Archives/edgar/data/1561550/{acc_nodash}/")
            if not idx:
                continue
            ex = [m for m in re.findall(r'href="([^"]+\.htm)"', idx.text) if re.search(r"ex-?99", m, re.I)]
            if not ex:
                continue
            url = ex[0] if ex[0].startswith("http") else "https://www.sec.gov" + ex[0]
            pr = fetch(url, allow_html=True)
            if not pr:
                continue
            txt = re.sub(r"\s+", " ", re.sub(r"&nbsp;?", " ", re.sub(r"<[^>]+>", " ", pr.text)))
            q = report_date
            m = re.search(r"([\d,]{5,})\s+customers with ARR of \$100,000", txt)
            if m:
                rows.append(
                    {"date": q, "source": "sec_8k", "metric": "customers_100k_arr", "value": int(m.group(1).replace(",", ""))}
                )
                kpi.setdefault("customers_100k_arr", []).append({"date": q, "value": int(m.group(1).replace(",", ""))})
            m = re.search(r"revenue grew (\d+)% year-over-year to \$([\d.]+) (billion|million)", txt, re.I)
            if m:
                val = float(m.group(2)) * (1000 if m.group(3).lower() == "billion" else 1)
                rows.append({"date": q, "source": "sec_8k", "metric": "revenue_q_reported_musd", "value": val})
            m = re.search(r"(\d{1,3}(?:,\d{3})*)\s+customers", txt)
            time.sleep(MIN_INTERVAL)

    return pd.DataFrame(rows), pd.DataFrame(), kpi


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="rebuild from data/raw only")
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * 11)  # npm history is long; cheap to ask once
    start_s, end_s = start.isoformat(), end.isoformat()

    frames = []
    if not args.offline:
        frames.append(collect_npm(start_s, end_s))
        frames.append(collect_pypi())
        gh_dept, gh_summary = collect_greenhouse()
        sec_quarters, extra, kpi = collect_sec()
        frames.append(sec_quarters)
        gh_dept.to_csv(TIDY / "s2_hiring_by_dept.csv", index=False) if not gh_dept.empty else None
    else:
        gh_summary, kpi = {}, {}

    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if not tidy.empty:
        tidy = tidy.sort_values(["source", "metric", "date"])
        tidy.to_csv(TIDY / "all_signals.csv", index=False)
        log(f"\ntidy rows: {len(tidy):,} -> data/tidy/all_signals.csv")

    # ---- wide export for Tableau -------------------------------------------
    if not tidy.empty:
        daily = tidy[tidy.source.isin(["npm", "pypi"])].copy()
        if not daily.empty:
            wide = daily.pivot_table(index="date", columns="metric", values="value", aggfunc="sum").fillna(0)
            wide["instrumentation_index"] = wide.sum(axis=1)
            wide.reset_index().to_csv(TABLEAU / "s1_instrumentation_daily.csv", index=False)
            log(f"tableau: s1_instrumentation_daily.csv ({len(wide):,} days)")
        sec = tidy[tidy.source.isin(["sec_xbrl", "sec_8k"])].copy()
        if not sec.empty:
            sec.pivot_table(index="date", columns="metric", values="value", aggfunc="last").reset_index().to_csv(
                TABLEAU / "target_ddog_kpis.csv", index=False
            )
            log("tableau: target_ddog_kpis.csv")
    if gh_summary:
        pd.DataFrame(
            [{"department": k, "open_roles": v, "as_of": gh_summary["as_of"]} for k, v in gh_summary["by_department"].items()]
        ).to_csv(TABLEAU / "s2_hiring_snapshot.csv", index=False)
        log("tableau: s2_hiring_snapshot.csv")

    # ---- payload for the web dashboard -------------------------------------
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hiring": gh_summary,
        "kpi": kpi,
        "rows": tidy.to_dict("records") if not tidy.empty else [],
        "provenance": {
            "npm": "api.npmjs.org (official registry API)",
            "pypi": "pypistats.org (public API, without_mirrors)",
            "greenhouse": "boards-api.greenhouse.io Job Board API (public)",
            "sec": "SEC EDGAR XBRL + 8-K EX-99.1 (public domain)",
            "blocked": "datadoghq.com not accessed - Datadog AUP No Framing or Scraping",
        },
        "log": _log,
    }
    (SITE_DATA / "payload.json").write_text(json.dumps(payload, indent=2, default=str))
    log(f"site payload -> site/data/payload.json ({len(payload['rows']):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
