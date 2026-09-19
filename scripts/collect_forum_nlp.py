"""Collect public developer-community mentions of Datadog for dashboard NLP signals.

This collector intentionally uses documented/public APIs only.  It does not
scrape Google result pages or Datadog-owned properties.  Each source response
is retained under data/raw/ and the dashboard receives aggregates, not posts.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import PROCESSED_DIR, get_json, save_raw, write_table  # noqa: E402

START = date(2018, 1, 1)
HN_URL = "https://hn.algolia.com/api/v1/search_by_date"
STACK_URL = "https://api.stackexchange.com/2.3/search/advanced"
DETAIL_COLUMNS = [
    "date", "source", "record_id", "title", "text", "url", "score", "responses",
    "positive", "negative", "incident", "evaluation", "adoption", "pricing", "legal_basis",
]

TOPICS = {
    "incident": ("outage", "incident", "downtime", "error", "bug", "broken", "fail", "alert"),
    "evaluation": ("compare", "comparison", "alternative", "migrate", "migration", "evaluate", " vs "),
    "adoption": ("install", "integration", "integrate", "setup", "deploy", "instrument", "sdk", "trace"),
    "pricing": ("pricing", "price", "cost", "expensive", "billing", "bill"),
}
POSITIVE = ("love", "great", "good", "excellent", "impressed", "recommend", "easy", "fast")
NEGATIVE = ("outage", "incident", "error", "bug", "expensive", "slow", "issue", "problem", "broken", "fail")


def clean_text(value: object) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def annotate(text: str) -> dict[str, int]:
    normalized = f" {text.lower()} "
    return {
        "positive": int(any(term in normalized for term in POSITIVE)),
        "negative": int(any(term in normalized for term in NEGATIVE)),
        **{name: int(any(term in normalized for term in terms)) for name, terms in TOPICS.items()},
    }


def chunks(start: date, end: date, days: int):
    cursor = start
    while cursor <= end:
        finish = min(end, cursor + timedelta(days=days - 1))
        yield cursor, finish
        cursor = finish + timedelta(days=1)


def epoch(day: date, *, end: bool = False) -> int:
    point = datetime.combine(day + timedelta(days=1 if end else 0), datetime.min.time(), tzinfo=timezone.utc)
    return int(point.timestamp()) - (1 if end else 0)


def hn_rows(start: date, end: date) -> list[dict]:
    rows: list[dict] = []
    for left, right in chunks(start, end, 365):
        page = 0
        while True:
            params = {
                "query": "datadog", "tags": "(story,comment)", "numericFilters":
                f"created_at_i>={epoch(left)},created_at_i<={epoch(right, end=True)}",
                "hitsPerPage": 1000, "page": page,
            }
            url = f"{HN_URL}?{urlencode(params)}"
            payload = get_json(url, source="hackernews", note="Datadog public discussion search")
            save_raw("hackernews", f"datadog_{left:%Y%m%d}_{right:%Y%m%d}_{page}", payload, url=url,
                     note="Datadog discussion search; stories and comments")
            for hit in payload.get("hits", []):
                text = clean_text(" ".join(filter(None, [hit.get("title"), hit.get("story_title"), hit.get("comment_text")])))
                if not text:
                    continue
                rows.append({
                    "date": str(hit.get("created_at", ""))[:10], "source": "Hacker News",
                    "record_id": str(hit.get("objectID")), "title": clean_text(hit.get("title") or hit.get("story_title")),
                    "text": text, "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('story_id') or hit.get('objectID')}",
                    "score": hit.get("points") or 0, "responses": hit.get("num_comments") or 0,
                    "legal_basis": "Hacker News public search API (Algolia index)", **annotate(text),
                })
            if page + 1 >= int(payload.get("nbPages", 0)):
                break
            page += 1
    return rows


def stack_rows(start: date, end: date) -> list[dict]:
    rows: list[dict] = []
    for left, right in chunks(start, end, 365):
        page = 1
        while True:
            params = {"site": "stackoverflow", "tagged": "datadog", "fromdate": epoch(left),
                      "todate": epoch(right, end=True), "pagesize": 100, "page": page,
                      "order": "asc", "sort": "creation", "filter": "withbody"}
            url = f"{STACK_URL}?{urlencode(params)}"
            payload = get_json(url, source="stackexchange", note="Stack Overflow questions tagged datadog")
            save_raw("stackexchange", f"datadog_{left:%Y%m%d}_{right:%Y%m%d}_{page}", payload, url=url,
                     note="Stack Overflow public questions tagged datadog")
            for item in payload.get("items", []):
                text = clean_text(f"{item.get('title', '')} {item.get('body', '')}")
                rows.append({
                    "date": datetime.fromtimestamp(item["creation_date"], tz=timezone.utc).date().isoformat(),
                    "source": "Stack Overflow", "record_id": str(item["question_id"]),
                    "title": clean_text(item.get("title")), "text": text, "url": item.get("link", ""),
                    "score": item.get("score") or 0, "responses": item.get("answer_count") or 0,
                    "legal_basis": "Stack Exchange public API (CC BY-SA content; aggregated analysis only)",
                    **annotate(text),
                })
            if not payload.get("has_more"):
                break
            if payload.get("backoff"):
                time.sleep(int(payload["backoff"]))
            page += 1
    return rows


def source_start(existing: pd.DataFrame, source: str, requested: date) -> date:
    prior = existing[existing["source"] == source] if not existing.empty else existing
    if prior.empty:
        return requested
    newest = pd.to_datetime(prior["date"], errors="coerce").max()
    return max(requested, newest.date() - timedelta(days=7)) if pd.notna(newest) else requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    args = parser.parse_args()
    detail_path = PROCESSED_DIR / "forum_mentions_detail.csv"
    existing = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame(columns=DETAIL_COLUMNS)
    today = datetime.now(timezone.utc).date()
    fresh = []
    fresh.extend(hn_rows(source_start(existing, "Hacker News", args.start), today))
    fresh.extend(stack_rows(source_start(existing, "Stack Overflow", args.start), today))
    detail = pd.concat([existing, pd.DataFrame(fresh)], ignore_index=True)
    detail = detail.reindex(columns=DETAIL_COLUMNS).dropna(subset=["date", "record_id"])
    detail = detail.drop_duplicates(["source", "record_id"], keep="last").sort_values(["date", "source", "record_id"])
    for column in ["score", "responses", "positive", "negative", "incident", "evaluation", "adoption", "pricing"]:
        detail[column] = pd.to_numeric(detail[column], errors="coerce").fillna(0).astype(int)
    daily = detail.groupby(["date", "source", "legal_basis"], as_index=False).agg(
        mentions=("record_id", "count"), score_sum=("score", "sum"), responses_sum=("responses", "sum"),
        positive_mentions=("positive", "sum"), negative_mentions=("negative", "sum"),
        incident_mentions=("incident", "sum"), evaluation_mentions=("evaluation", "sum"),
        adoption_mentions=("adoption", "sum"), pricing_mentions=("pricing", "sum"),
    )
    write_table(detail, "forum_mentions_detail")
    write_table(daily, "forum_mentions_daily")
    print(f"forum detail={len(detail)} daily={len(daily)} new={len(fresh)}")


if __name__ == "__main__":
    main()
