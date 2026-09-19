# DDOG Alternative Data Dashboard

An alternative-data research prototype for tracking Datadog (NASDAQ: DDOG)
ahead of quarterly earnings. It combines public developer-activity, cloud,
macro, competitive, and company signals with quarterly financial targets.

## What is included

- Public-data collection and transformation scripts
- Walk-forward backtesting and current-quarter forecast logic
- A bilingual, browser-based dashboard with factor diagnostics and configurable
  model inputs
- Docker and system-service templates for a local deployment

## What is deliberately excluded

This public repository does **not** include credentials, server addresses,
deployment domain configuration, raw API snapshots, processed forum text,
generated dashboard payloads, or Tableau workbooks. Those materials may carry
private configuration, third-party content, or environment-specific paths.

## Local setup

1. Create a Python environment and install `deploy/requirements.txt`.
2. Copy `.env.example` to `.env` and use local database credentials.
3. Start MySQL with `docker compose -f deploy/docker-compose.yml up -d`.
4. Run the collection and processing scripts under `scripts/` to build local
   datasets from their documented public sources.
5. Load the generated tables with `scripts/sync_mysql.py` and start the API
   with `deploy/ddog_api.py`.

The dashboard expects the API to serve `/payload` and `/forecast` endpoints.
It is designed for local research use and should be reviewed before any
production deployment.

## Data and compliance

The research workflow is intended for publicly accessible or legally
accessible sources only. Data collection should be run with respect for source
terms, rate limits, and applicable licenses. No material non-public
information is used.
