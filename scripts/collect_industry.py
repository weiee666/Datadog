"""Public industry-activity collectors, kept separate from Datadog company signals.

The data measures the broader observability and cloud-infrastructure ecosystem:
open-source package adoption, container image pulls, cloud provider deployment
tooling, and technical-community activity.  It never reuses a Datadog package as
an industry proxy.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_s1 import CHUNK_DAYS, OVERLAP_DAYS, _chunks, load_existing, upsert  # noqa: E402
from fetch_utils import LEGAL_BASIS, PROCESSED_DIR, NotFound, get_json, save_raw, write_table  # noqa: E402

NPM_START = "2018-01-01"
NPM_PACKAGES = {
    "observability": [
        "@opentelemetry/api", "@opentelemetry/sdk-node",
        "@opentelemetry/auto-instrumentations-node", "prom-client",
    ],
    "cloud_infrastructure": ["@kubernetes/client-node", "@pulumi/pulumi"],
}
PYPI_PACKAGES = {
    "observability": ["opentelemetry-api", "opentelemetry-sdk", "prometheus-client"],
    "cloud_infrastructure": ["kubernetes", "boto3"],
}
DOCKER_REPOS = {
    "observability": ["otel/opentelemetry-collector", "prom/prometheus", "grafana/grafana", "jaegertracing/all-in-one"],
}
TERRAFORM_PROVIDERS = {
    "cloud_infrastructure": ["hashicorp/aws", "hashicorp/azurerm", "hashicorp/google"],
}
DAILY_CSV = PROCESSED_DIR / "industry_activity_daily.csv"
CUM_CSV = PROCESSED_DIR / "industry_activity_cumulative_snapshots.csv"


def row(day: str, series_id: str, value, unit: str, source: str, category: str) -> dict:
    return {"date": day, "series_id": series_id, "value": value, "unit": unit,
            "freq": "D", "source": source, "category": category,
            "legal_basis": LEGAL_BASIS[source]}


def collect_npm(existing: pd.DataFrame, only_packages: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    today = datetime.now(timezone.utc).date()
    for category, packages in NPM_PACKAGES.items():
        for package in packages:
            if only_packages and package not in only_packages:
                continue
            series_id = f"industry.npm.{package}.downloads"
            have = existing[existing["series_id"] == series_id]
            start = (pd.to_datetime(have["date"].max()).date() - timedelta(days=OVERLAP_DAYS)
                     if len(have) else datetime.strptime(NPM_START, "%Y-%m-%d").date())
            print(f"  npm industry: {package} ({category})")
            for begin, end in _chunks(start, today, CHUNK_DAYS):
                url = f"https://api.npmjs.org/downloads/range/{begin}:{end}/{quote(package, safe='')}"
                try:
                    payload = get_json(url, source="npm", note=f"industry npm {package} {begin}..{end}")
                except NotFound:
                    continue
                except Exception as exc:
                    print(f"    ! {begin}..{end}: {str(exc)[:70]}")
                    continue
                days = payload.get("downloads") or []
                if days:
                    save_raw("npm", f"industry_{package.replace('/', '_').replace('@', '')}_{begin}_{end}", payload,
                             url=url, note=f"industry npm {package}")
                    rows.extend(row(x["day"], series_id, x["downloads"], "downloads/day", "npm", category)
                                for x in days)
    return pd.DataFrame(rows)


def collect_pypi() -> pd.DataFrame:
    rows: list[dict] = []
    for category, packages in PYPI_PACKAGES.items():
        for package in packages:
            url = f"https://pypistats.org/api/packages/{package}/overall"
            try:
                payload = get_json(url, source="pypistats", note=f"industry PyPI {package}")
            except Exception as exc:
                print(f"  ! pypi {package}: {str(exc)[:70]}")
                continue
            save_raw("pypistats", f"industry_{package}_overall", payload, url=url, note=f"industry PyPI {package}")
            rows.extend(row(item["date"], f"industry.pypi.{package}.downloads", item["downloads"],
                            "downloads/day", "pypistats", category)
                        for item in payload.get("data", []) if item.get("category") == "without_mirrors")
    return pd.DataFrame(rows)


def collect_cumulative() -> pd.DataFrame:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows: list[dict] = []
    for category, repositories in DOCKER_REPOS.items():
        for repository in repositories:
            url = f"https://hub.docker.com/v2/repositories/{repository}/"
            try:
                payload = get_json(url, source="dockerhub", note=f"industry Docker Hub {repository}")
            except Exception as exc:
                print(f"  ! docker {repository}: {str(exc)[:70]}")
                continue
            name = repository.replace("/", "_")
            save_raw("dockerhub", f"industry_{name}", payload, url=url, note=f"industry Docker Hub {repository}")
            rows.append(row(today, f"industry.dockerhub.{repository}.cum_pulls", payload.get("pull_count"),
                            "cumulative pulls", "dockerhub", category))
    for category, providers in TERRAFORM_PROVIDERS.items():
        for provider in providers:
            url = f"https://registry.terraform.io/v1/providers/{provider}"
            try:
                payload = get_json(url, source="terraform", note=f"industry Terraform {provider}")
            except Exception as exc:
                print(f"  ! terraform {provider}: {str(exc)[:70]}")
                continue
            save_raw("terraform", f"industry_{provider.replace('/', '_')}", payload, url=url,
                     note=f"industry Terraform {provider}")
            rows.append(row(today, f"industry.terraform.{provider}.cum_downloads", payload.get("downloads"),
                            "cumulative downloads", "terraform", category))
    return pd.DataFrame(rows)


def drop_incomplete_npm_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove API placeholder days where the entire industry npm basket is zero.

    npm can return a zero for the in-progress UTC day before its download logs
    settle.  A real package can be quiet, but every package in both baskets
    simultaneously being zero is an incomplete-source observation, not demand.
    """
    npm = frame[frame["series_id"].str.startswith("industry.npm.")].copy()
    totals = npm.groupby("date", dropna=False)["value"].sum()
    bad_days = totals[totals == 0].index.tolist()
    if bad_days:
        print(f"  删除 {len(bad_days)} 个行业 npm 全零未结算日：{', '.join(map(str, bad_days[-5:]))}")
        return frame[~((frame["series_id"].str.startswith("industry.npm.")) & frame["date"].isin(bad_days))].copy()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-only", action="store_true", help="Only refresh cumulative Docker/Terraform snapshots")
    parser.add_argument("--npm-package", action="append", default=[],
                        help="Collect one npm package at a time; repeatable, for resumable historical backfills")
    parser.add_argument("--skip-npm", action="store_true", help="Refresh PyPI and cumulative sources only")
    args = parser.parse_args()
    existing = load_existing(DAILY_CSV)
    if args.snapshots_only:
        # A snapshot refresh must never touch the historical daily table.
        daily = existing
    else:
        npm = (pd.DataFrame() if args.skip_npm
               else collect_npm(existing, set(args.npm_package) or None))
        pypi = collect_pypi()
        new_daily = pd.concat([npm, pypi], ignore_index=True)
        daily = drop_incomplete_npm_days(upsert(existing, new_daily))
    cumulative = upsert(load_existing(CUM_CSV), collect_cumulative())
    for path in write_table(daily, "industry_activity_daily"):
        print("  ->", path)
    for path in write_table(cumulative, "industry_activity_cumulative_snapshots"):
        print("  ->", path)
    print(f"industry daily: {len(daily):,} rows / {daily['series_id'].nunique() if len(daily) else 0} series")


if __name__ == "__main__":
    main()
