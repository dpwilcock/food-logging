# Food Logging Pipeline

Part of my Ai/LLM Agent setup - allows verbally talking to telegram, whisper transcription, and recording the items of food I've eaten.
Nightly run uses the LLM to pull in calorie/kJ count for the food either from a cache, or from the internet at large.
Works great - lowest friction food tracking, so that for the most part it's easy to just do it and get on with life.

Parse and store food/drink intake from raw text, estimate energy (kJ) nightly, and report.

Parses free-text food and drink log entries (e.g. "lunch: chicken sandwich and a coffee"), normalizes number-words, infers meal occasion, stores to PostgreSQL, and estimates energy intake using a food database.

## Components

- `food_ingest.py` — parse a raw log line and insert a `food_log_entry`
- `food_estimate.py` — nightly estimation pass: fill missing kJ values from known-food lookups

## Requirements

- Python 3.9+
- `pip install psycopg2-binary`

## Configuration (environment variables)

| Variable | Purpose |
|----------|---------|
| `PGDATABASE` | PostgreSQL database name (default: `postgres`) |

## Usage

```bash
python3 food_ingest.py "ate a chicken sandwich and a coffee" [--source-msg-id 123] [--input-mode text|voice]
python3 food_estimate.py            # nightly kJ backfill/estimation
```

`food_ingest.py` returns JSON with the inserted entry id and parsed fields (occasion, planned/unplanned, consumed_at), ready for the caller to confirm back to the user.

## Schema notes

Expects a `food_log_entry` table with at minimum: raw text, parsed items, occasion type, consumed_at timestamp, and a kJ column filled by the estimator.
