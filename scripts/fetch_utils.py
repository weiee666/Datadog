"""合规数据抓取工具 —— 所有采集脚本的唯一出口（single choke point）。

合规基线，见 docs/data_source_strategy.md 第 3 节：
  1. 声明可联系的 User-Agent（SEC 明确要求，否则会被 "Undeclared Automated Tool" 拦截）；
  2. 按 host 限速（默认 >=1.2 秒；SEC 官方上限 10 req/s，我们远低于）；
  3. 域名黑名单：任何 *.datadoghq.com 一律拒绝发请求
     —— Datadog Acceptable Use Policy "No Framing or Scraping" 明文禁止
        "any robot, spider, ... or other manual or automatic device to retrieve,
         index, scrape, data mine, or in any way gather any ... content from the
         Service or Site"，且适用域包含 www.datadoghq.com 及其子域。
     本作业由 Datadog 出题，故采取最保守立场：连官方 RSS 也不抓。
  4. 每个响应原样落盘 data/raw/，保留审计线索（可复现 + 可自证合规）。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

UA = "DDOG-altdata-research/0.1 (take-home research; contact: analyst@example.com)"

# --- 合规黑名单：Datadog 自有站点全家族（AUP "No Framing or Scraping"）---------
BLOCKED_HOST_SUFFIXES = (".datadoghq.com", "datadoghq.com", ".ddog-gov.com", "ddog-gov.com")

# --- 每个来源的合规依据，写进输出的 legal_basis 列，可直接用于报告 -------------
LEGAL_BASIS = {
    "npm": "npm registry public downloads API (third-party registry, not Datadog property)",
    "pypistats": "pypistats.org public API, derived from PyPI public download logs",
    "terraform": "Terraform Registry public API (HashiCorp, third-party)",
    "dockerhub": "Docker Hub public API (third-party)",
    "stackexchange": "Stack Exchange public API (CC BY-SA content; aggregated tag counts only)",
    "hackernews": "Hacker News public search API (Algolia index; public discussion content, aggregated analysis only)",
    "greenhouse": "Greenhouse Job Board API - documented public interface for job syndication",
    "dol": "U.S. Dept. of Labor OFLC public disclosure data (U.S. government public record)",
    "sec": "SEC EDGAR / XBRL public API - programmatic access permitted (<=10 req/s, declared UA)",
    "crt": "crt.sh public Certificate Transparency log search",
    "github": "GitHub public REST API (open-source repositories with explicit licenses)",
    "nasdaq": "Nasdaq public quote API (market data, third-party)",
    "fred": ("FRED public CSV download endpoint (fredgraph.csv), Federal Reserve Bank of "
             "St. Louis - latest-vintage values; FRED cited as source; personal/non-commercial "
             "research use; NOT used for ML training; only public-domain series "
             "(BEA/BLS/Census/Fed) selected to avoid third-party copyright notices"),
    "alfred": ("ALFRED archival CSV endpoint (alfredgraph.csv), Federal Reserve Bank of "
               "St. Louis - point-in-time (as-of vintage) values used to eliminate "
               "revision/look-ahead bias in backtests"),
    "worldbank": "World Bank Open Data API, indicator NY.GDP.MKTP.KD.ZG (CC BY 4.0)",
}

MIN_INTERVAL = 1.2  # 同 host 最小请求间隔（秒）
_last_hit: dict[str, float] = {}
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": UA, "Accept": "application/json,text/html,application/xml,*/*"}
)


class ComplianceError(RuntimeError):
    """请求被本地合规基线拦截。"""


class NotFound(Exception):
    """目标不存在（HTTP 404）——不重试，由调用方自行跳过。

    典型场景：npm 在包首次发布之前的日期区间会返回 404（并非错误）。
    """


def _check_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if any(host == s.lstrip(".") or host.endswith(s) for s in BLOCKED_HOST_SUFFIXES):
        raise ComplianceError(
            f"已拦截对 Datadog 自有站点的请求：{host}\n"
            f"原因：Datadog Acceptable Use Policy 'No Framing or Scraping'。\n"
            f"若确需该数据，请改用第三方档案（Common Crawl / HTTP Archive）或开源仓库。"
        )
    return host


def _throttle(host: str) -> None:
    wait = MIN_INTERVAL - (time.time() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.time()


def fetch(url: str, *, source: str, note: str = "", tries: int = 3, timeout: int = 60) -> requests.Response:
    """带合规检查 + 限速 + 重试的 GET。"""
    host = _check_host(url)
    last_exc: Exception | None = None
    for attempt in range(1, tries + 1):
        _throttle(host)
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                raise NotFound(url)
            if r.status_code in (429, 500, 502, 503, 504):  # 可重试
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2.0 * attempt)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"fetch failed after {tries} tries: {url} :: {last_exc}")


def get_json(url: str, *, source: str, note: str = "", **kw):
    r = fetch(url, source=source, note=note, **kw)
    return r.json()


def save_raw(source: str, name: str, payload, *, url: str, note: str = "", day: str | None = None) -> Path:
    """原样落盘一份原始快照，附抓取元数据，保证可审计/可复现。"""
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = RAW_DIR / source / day
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    blob = json.dumps(
        {
            "meta": {
                "source": source,
                "name": name,
                "url": url,
                "note": note,
                "legal_basis": LEGAL_BASIS.get(source, "n/a"),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "user_agent": UA,
            },
            "payload": payload,
        },
        indent=2,
        ensure_ascii=False,
    )
    path.write_text(blob, encoding="utf-8")
    return path


def write_table(df, name: str, *, also_parquet: bool = True) -> list[Path]:
    """写出处理后的表：CSV（utf-8-sig，Excel 直接双击可开）+ Parquet（供程序快速读取）。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    csv_path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    out.append(csv_path)
    if also_parquet:
        pq_path = PROCESSED_DIR / f"{name}.parquet"
        try:
            df.to_parquet(pq_path, index=False)
            out.append(pq_path)
        except Exception as e:  # pyarrow 缺失时不致命
            print(f"   (parquet skipped for {name}: {e})")
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
