"""Collect peer SDK adoption proxies and quarterly reported revenue.

The npm series are official Node.js instrumentation packages. They proxy
developer adoption rather than company-wide usage, so they are kept separate
from reported financial results.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
import argparse

import pandas as pd

from collect_s1 import CHUNK_DAYS, OVERLAP_DAYS, _chunks, upsert
from fetch_utils import LEGAL_BASIS, PROCESSED_DIR, NotFound, get_json, save_raw, write_table


PEERS = {
    "Datadog": {"cik": "0001561550", "package": "dd-trace"},
    "Dynatrace": {"cik": "0001773383", "package": "@dynatrace/oneagent"},
    "Elastic": {"cik": "0001707753", "package": "elastic-apm-node"},
}
DAILY_PATH = PROCESSED_DIR / "competitor_npm_daily.csv"
REVENUE_PATH = PROCESSED_DIR / "competitor_revenue_quarterly.csv"
NPM_START = date(2018, 1, 1)


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["date", "company", "package", "value", "unit", "freq", "source", "legal_basis"])


def quarter_label(day: str) -> str:
    return f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}"


def collect_npm(backfill: bool = False) -> pd.DataFrame:
    existing = load_existing(DAILY_PATH)
    rows: list[dict] = []
    today = datetime.now(timezone.utc).date()
    for company, meta in PEERS.items():
        package = meta["package"]
        have = existing[existing["company"] == company]
        start = (NPM_START if backfill else pd.to_datetime(have["date"].max()).date() - timedelta(days=OVERLAP_DAYS)
                 if len(have) else NPM_START)
        print(f"  npm peer: {company} / {package}")
        for begin, end in _chunks(start, today, CHUNK_DAYS):
            url = f"https://api.npmjs.org/downloads/range/{begin}:{end}/{quote(package, safe='')}"
            try:
                payload = get_json(url, source="npm", note=f"peer npm {company} {package} {begin}..{end}")
            except NotFound:
                continue
            except Exception as exc:
                print(f"    ! {begin}..{end}: {str(exc)[:70]}")
                continue
            points = payload.get("downloads") or []
            if not points:
                continue
            save_raw("npm", f"peer_{company.lower()}_{package.replace('/', '_').replace('@', '')}_{begin}_{end}", payload,
                     url=url, note=f"peer npm downloads: {company} / {package}")
            rows.extend({"date": point["day"], "company": company, "package": package,
                         "value": point["downloads"], "unit": "downloads/day", "freq": "D",
                         "source": "npm", "legal_basis": LEGAL_BASIS["npm"]} for point in points)
    new = pd.DataFrame(rows)
    if not len(new):
        return existing
    result = (pd.concat([existing, new], ignore_index=True)
              .drop_duplicates(["date", "company", "package"], keep="last")
              .sort_values(["company", "date"]).reset_index(drop=True))
    latest = result["date"].max()
    if result.loc[result["date"] == latest, "value"].sum() == 0:
        result = result[result["date"] != latest].copy()
    return result


def company_revenue(company: str, cik: str) -> list[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = get_json(url, source="sec", note=f"peer revenue companyfacts {company}")
    save_raw("sec_peers", f"{company.lower()}_companyfacts", facts, url=url, note=f"peer quarterly revenue: {company}")
    gaap = facts.get("facts", {}).get("us-gaap", {})
    concept = next((gaap.get(tag) for tag in ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues") if gaap.get(tag)), None)
    if not concept:
        raise RuntimeError(f"no revenue concept for {company}")
    units = concept.get("units", {}).get("USD", [])
    quarters, annual = {}, {}
    for unit in units:
        if unit.get("form") not in ("10-Q", "10-K") or not unit.get("start"):
            continue
        start, end = date.fromisoformat(unit["start"]), date.fromisoformat(unit["end"])
        duration = (end - start).days
        if 80 <= duration <= 100:
            if unit.get("filed", "") >= quarters.get(unit["end"], {}).get("filed", ""):
                quarters[unit["end"]] = unit
        elif unit.get("form") == "10-K" and 360 <= duration <= 370:
            annual[unit["end"]] = unit
    values = {end: float(unit["val"]) for end, unit in quarters.items()}
    for end, unit in annual.items():
        if end not in values:
            year = end[:4]
            prior = [values.get(f"{year}-{month}-31") for month in ("03", "06", "09")]
            if all(value is not None for value in prior):
                values[end] = float(unit["val"]) - sum(prior)
    out = []
    for end, revenue in sorted(values.items()):
        out.append({"company": company, "quarter": quarter_label(end), "period_end": end, "revenue": revenue,
                    "source": "sec", "legal_basis": LEGAL_BASIS["sec"]})
    return out


def collect_revenue() -> pd.DataFrame:
    rows = []
    for company, meta in PEERS.items():
        print(f"  SEC peer revenue: {company}")
        rows.extend(company_revenue(company, meta["cik"]))
    frame = pd.DataFrame(rows).sort_values(["company", "period_end"])
    frame["revenue_yoy_pct"] = frame.groupby("company")["revenue"].pct_change(4) * 100
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Refetch npm history from 2018")
    args = parser.parse_args()
    print("== Peer competitor data ==")
    daily = collect_npm(args.backfill)
    revenue = collect_revenue()
    for path in write_table(daily, "competitor_npm_daily"):
        print("  ->", path)
    for path in write_table(revenue, "competitor_revenue_quarterly"):
        print("  ->", path)


if __name__ == "__main__":
    main()
