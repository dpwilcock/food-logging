#!/usr/bin/env python3
f"""
food_estimate.py — Nightly batch: split entries into items, match against cache,
estimate novel foods via one model call, write items + update cache.

Usage:
  python3 food_estimate.py [--dry-run] [--date YYYY-MM-DD]

Exit code 0: success (or nothing to do).
Exit code 1: error.

The model call is provided by the agent harness (OpenClaw): this script prepares
novel items as a JSON batch for the agent to estimate (see the `estimate` block in
main()); the agent returns kJ values that this script writes back. The agent-side
prompt lives with the harness skill, not in this repo.
"""

import os
import re
import sys
import json
import argparse
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

SCHEMA = os.environ.get("PGSCHEMA", "DW Health")

DB = os.environ.get("PGDATABASE", "postgres")
TZ = ZoneInfo("America/Vancouver")  # DST-aware

# ── ITEM SPLITTING ────────────────────────────────────────────────────

# Conjunction markers to split on
SPLIT_RE = re.compile(
    r'\s+(?:and|&|plus|\+)\s+|'   # "coffee and toast" → two items
    r'\s*,\s*|'                    # "coffee, toast" → two
    r'\s+with\s+'                  # "coffee with milk" → single item, keep together
    # Actually "with" should NOT split — "coffee with milk" is one item.
    # Override: only split on "and", "&", "+", ","
)

# Phrases that should stay together (don't split on "and" inside them)
KEEP_TOGETHER = [
    "fish and chips",
    "peanut butter and jam",
    "peanut butter and jelly",
    "mac and cheese",
    "macaroni and cheese",
    "rice and beans",
    "ham and cheese sandwich",
    "crackers and cheese",
    "hummus and pita",
    "bangers and mash",
]

def split_items(text: str) -> list[str]:
    """
    Split raw_text into food items.
    Respects KEEP_TOGETHER phrases (don't split those).
    """
    # Temporarily replace keep-together phrases
    replacements = {}
    for i, phrase in enumerate(KEEP_TOGETHER):
        placeholder = f"__KEEP_{i}__"
        text_lower = text.lower()
        if phrase in text_lower:
            idx = text_lower.index(phrase)
            replacements[placeholder] = text[idx:idx + len(phrase)]
            text = text[:idx] + placeholder + text[idx + len(phrase):]

    # Split on commas, "and", "&" (but not our placeholders)
    parts = re.split(r'\s*,\s*|\s+(?:and|&)\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    # Restore KEEP_TOGETHER phrases
    for placeholder, original in replacements.items():
        parts = [p.replace(placeholder, original) for p in parts]

    return parts

# ── FOOD KEY NORMALIZATION (mirrors ingest script) ────────────────────

STOP_WORDS = {"a", "an", "some", "of", "with", "my", "the"}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "half": 0.5, "quarter": 0.25,
}

UNITS = {"slice", "slices", "piece", "pieces", "cup", "cups", "bowl", "bowls",
         "serving", "servings", "handful", "handfuls", "glass", "glasses",
         "can", "cans", "bottle", "bottles", "bar", "bars", "scoop", "scoops",
         "plate", "plates", "egg", "eggs", "cookie", "cookies",
         "roll", "rolls", "mug", "mugs", "oz", "g", "ml", "l",
         "tsp", "tbsp", "lb", "kg", "packet", "pack", "bag", "box"}

def extract_qty_unit(text: str) -> tuple[float, str, str]:
    """
    Extract quantity, unit, and normalized food_key from an item string.
    Returns (qty, unit, food_key).
    """
    text = text.strip()
    qty = 1.0
    unit = "serving"
    food_part = text

    # Try extracting leading number
    m = re.match(r'(\d+(?:\.\d+)?)\s*(.*)', text)
    if m:
        qty = float(m.group(1))
        rest = m.group(2).strip()
    else:
        # Try number words at start
        words = text.split()
        if words:
            first = words[0].lower().strip(",.;:!?")
            if first in NUMBER_WORDS:
                val = NUMBER_WORDS[first]
                qty = float(val)
                rest = " ".join(words[1:])
            else:
                rest = text
        else:
            rest = text

    # Extract unit from start of rest
    rest_words = rest.split()
    if rest_words:
        candidate = rest_words[0].lower().strip(",.;:!?")
        if candidate in UNITS:
            unit = candidate
            food_part = " ".join(rest_words[1:])
        else:
            food_part = rest
    else:
        food_part = rest

    # Normalize food_key
    food_key = normalize_food_key(food_part)
    return qty, unit, food_key

def normalize_food_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    while words and words[0] in {str(v) for v in NUMBER_WORDS.values()}:
        words.pop(0)
    return " ".join(words) if words else text

# ── UNIT COMPATIBILITY ───────────────────────────────────────────────

# Units that scale linearly with amount (per-ml, per-g, ...)
VOL_UNITS = {"ml", "l", "oz", "fl oz", "g", "kg", "tsp", "tbsp", "lb"}

# Units that represent a portion/serving (do NOT scale with ml/g amounts)
SERVING_UNITS = {
    "serving", "servings", "bowl", "bowls", "plate", "plates", "scoop", "scoops",
    "slice", "slices", "piece", "pieces", "bar", "bars", "cookie", "cookies",
    "glass", "glasses", "mug", "mugs", "cup", "cups", "can", "cans",
    "bottle", "bottles", "roll", "rolls", "egg", "eggs", "handful", "handfuls",
    "bag", "box", "packet", "pack", "stick", "sticks", "square", "squares",
}

def units_compatible(hit_unit: str, item_unit: str) -> bool:
    """True if a cache row's unit can be applied to an item's unit."""
    if hit_unit == item_unit:
        return True
    hit_serving = hit_unit in SERVING_UNITS
    item_serving = item_unit in SERVING_UNITS
    hit_vol = hit_unit in VOL_UNITS
    item_vol = item_unit in VOL_UNITS
    # Same family: serving-ish with serving-ish, or per-volume with per-volume
    return (hit_serving and item_serving) or (hit_vol and item_vol)

# ── SANITY CHECKS ────────────────────────────────────────────────────

# Units that are typically per-volume (small increments).
# If kcal_per_unit is implausibly high for these, it's probably a
# per-serving value mislabeled as per-unit-volume.
LIQUID_UNIT_MAX = {
    "ml": 5.0,       # no beverage is >5 kcal/ml (oil is ~9, but we don't drink oil by the ml)
    "l": 500.0,      # >500 kcal/litre is per-serving territory
    "cup": 350.0,    # a cup of pure fat is ~1800 kcal, but we don't drink cups of oil
    "cups": 350.0,
    "glass": 350.0,
    "glasses": 350.0,
    "oz": 50.0,      # fl oz — pure oil is ~250 kcal/fl oz, beverages max ~35
    "fl oz": 50.0,
}

def validate_estimate(food_key: str, unit: str, kcal_per_unit: float):
    """
    Return an error message if the estimate fails sanity checks, else None.
    """
    # Check liquid-volume units against sane ceilings
    if unit.lower() in LIQUID_UNIT_MAX:
        ceiling = LIQUID_UNIT_MAX[unit.lower()]
        if kcal_per_unit > ceiling:
            return (
                f"{food_key}: {kcal_per_unit} kcal/{unit} exceeds sanity ceiling of "
                f"{ceiling} kcal/{unit}. Likely a per-serving value mislabeled as per-{unit}. "
                f"Re-estimate with correct per-volume kcal."
            )

    # General: if a single unit would exceed 4000 kcal at typical serving, flag it
    typical_serving = {"ml": 250, "l": 0.25, "cup": 1, "cups": 1,
                       "glass": 1, "glasses": 1, "oz": 8, "fl oz": 8}
    if unit.lower() in typical_serving:
        serving_kcal = kcal_per_unit * typical_serving[unit.lower()]
        if serving_kcal > 4000:
            return (
                f"{food_key}: typical serving ({typical_serving[unit.lower()]}{unit}) "
                f"would be {serving_kcal:.0f} kcal — implausible. "
                f"Check unit vs estimate."
            )

    return None

# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nightly food estimation batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be estimated, don't write")
    parser.add_argument("--date", help="Process only this date (default: all pending)")
    parser.add_argument("--write-cache", action="store_true",
                        help="Accept cache row JSON on stdin and write")
    parser.add_argument("--affected-dates",
                        help="With --write-cache: comma-separated dates to re-estimate after cache write")
    parser.add_argument("--rebuild-day", help="Rebuild food_day for this date")
    args = parser.parse_args()

    conn = psycopg2.connect(f"dbname={DB}")

    # ── Cache write mode ──
    if args.write_cache:
        try:
            cache_data = json.load(sys.stdin)
            with conn.cursor() as cur:
                for row in cache_data:
                    food_key = row["food_key"]
                    unit = row.get("unit", "serving")
                    kcal_per_unit = float(row["kcal_per_unit"])

                    # ── SANITY CHECK ──
                    err = validate_estimate(food_key, unit, kcal_per_unit)
                    if err:
                        print(json.dumps({"error": f"sanity check failed: {err}"}),
                              file=sys.stderr)
                        sys.exit(1)

                    cur.execute(
                        """INSERT INTO "{SCHEMA}"."food_estimate_cache"
                           (food_key, canonical_name, unit, kcal_per_unit,
                            confidence, estimate_source)
                           VALUES (%s, %s, %s, %s, %s, 'model')
                           ON CONFLICT (food_key)
                           DO UPDATE SET times_seen = "food_estimate_cache".times_seen + 1,
                                         last_seen_at = now()
                           WHERE "food_estimate_cache".verified = false""",
                        (food_key, row["canonical_name"], unit,
                         kcal_per_unit, row["confidence"]))
            conn.commit()
            written_count = len(cache_data)
            print(json.dumps({"written": written_count, "food_keys": [r["food_key"] for r in cache_data]}))

            if not args.affected_dates:
                return  # done — caller handles rest

            # ── AUTO-COMPLETE: after writing new cache rows, re-run the
            # batch for each affected date. Since the novel foods are now
            # cached, they'll match as cache hits and the pipeline completes
            # atomically within this single script call.
            affected = [d.strip() for d in args.affected_dates.split(",") if d.strip()]
            results = []
            for d in affected:
                results.append(_run_batch_for_date(conn, date.fromisoformat(d)))
            conn.commit()
            print(json.dumps({"status": "auto_complete", "results": results}))
            return
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        finally:
            conn.close()

    # ── Day rebuild mode ──
    if args.rebuild_day:
        rebuild_date = date.fromisoformat(args.rebuild_day)
        rebuild_day(conn, rebuild_date)
        conn.commit()
        conn.close()
        return

    # ── Standard batch mode ──
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Get pending entries
            if args.date:
                cur.execute("""
                    SELECT id, raw_text, consumed_date
                    FROM "{SCHEMA}"."food_log_entry"
                    WHERE parse_status = 'pending'
                      AND parse_attempts < 3
                      AND superseded_by IS NULL
                      AND consumed_date = %s
                    ORDER BY id
                """, (args.date,))
            else:
                cur.execute("""
                    SELECT id, raw_text, consumed_date
                    FROM "{SCHEMA}"."food_log_entry"
                    WHERE parse_status = 'pending'
                      AND parse_attempts < 3
                      AND superseded_by IS NULL
                    ORDER BY id LIMIT 200
                """)
            entries = cur.fetchall()

            if not entries:
                print(json.dumps({"status": "no_pending"}))
                return

            # 2. For each entry, split into items
            all_items = []  # (entry_id, seq, food_key, description, qty, unit)
            all_food_keys = set()
            affected_dates = set()

            for entry in entries:
                raw = entry["raw_text"]
                parts = split_items(raw)
                affected_dates.add(entry["consumed_date"])

                for seq, part in enumerate(parts, 1):
                    qty, unit, food_key = extract_qty_unit(part)
                    all_items.append((entry["id"], seq, food_key, part.strip(), qty, unit))
                    all_food_keys.add(food_key)

            # 3. Check cache hits
            cache_hits = {}
            if all_food_keys:
                cur.execute("""
                    SELECT food_key, canonical_name, unit, kcal_per_unit, confidence
                    FROM "{SCHEMA}"."food_estimate_cache"
                    WHERE food_key = ANY(%s)
                """, (list(all_food_keys),))
                for row in cur.fetchall():
                    cache_hits[row["food_key"]] = dict(row)

            # 4. Separate hits from misses
            cache_write = []
            item_writes = []  # (entry_id, seq, food_key, description, qty, unit, kcal, kj, confidence, source)
            misses = []
            miss_id_counter = 1

            for entry_id, seq, food_key, desc, qty, unit in all_items:
                if food_key in cache_hits:
                    hit = cache_hits[food_key]
                    if not units_compatible(hit["unit"], unit):
                        # Cache row unit can't be applied to this item's unit
                        # (e.g. per-ml kcal vs "1 serving", or per-serving kcal
                        # vs "50 ml"). Treat as a miss so the estimator
                        # re-estimates with the correct per-volume value.
                        misses.append({
                            "id": miss_id_counter,
                            "food_key": food_key,
                            "qty": qty,
                            "unit": unit,
                            "entry_id": entry_id,
                            "seq": seq,
                            "description": desc,
                        })
                        miss_id_counter += 1
                        continue
                    kcal = round(float(hit["kcal_per_unit"]) * float(qty))
                    kj = round(kcal * 4.184)
                    item_writes.append((entry_id, seq, food_key, desc, qty, unit,
                                       kcal, kj, hit["confidence"], 'cache'))
                else:
                    misses.append({
                        "id": miss_id_counter,
                        "food_key": food_key,
                        "qty": qty,
                        "unit": unit,
                        "entry_id": entry_id,
                        "seq": seq,
                        "description": desc,
                    })
                    miss_id_counter += 1

            if args.dry_run:
                result = {
                    "status": "dry_run",
                    "total_entries": len(entries),
                    "total_items": len(all_items),
                    "cache_hits": len(item_writes),
                    "cache_misses": len(misses),
                    "affected_dates": [d.isoformat() for d in affected_dates],
                }
                if misses:
                    result["batch_input"] = "\n".join(
                        f"{m['id']}|{m['food_key']}|{m['qty']}|{m['unit']}"
                        for m in misses
                    )
                    result["miss_details"] = misses
                print(json.dumps(result, indent=2))
                return

            # 5. Write cache hits first (so they're committed even if model call fails)
            with conn.cursor() as wcur:
                # Write items with cache hits
                for item in item_writes:
                    entry_id, seq, food_key, desc, qty, unit, kcal, kj, conf, source = item
                    wcur.execute("""
                        INSERT INTO "{SCHEMA}"."food_log_item"
                            (entry_id, seq, food_key, description, qty, unit,
                             kcal, kj, confidence, estimate_source, estimated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (entry_id, seq) DO UPDATE SET
                            food_key = EXCLUDED.food_key,
                            kcal = EXCLUDED.kcal,
                            kj = EXCLUDED.kj,
                            confidence = EXCLUDED.confidence,
                            estimate_source = EXCLUDED.estimate_source,
                            estimated_at = EXCLUDED.estimated_at
                    """, item)

                # Bump times_seen on hit cache rows
                for food_key in {item[2] for item in item_writes}:
                    wcur.execute("""
                        UPDATE "{SCHEMA}"."food_estimate_cache"
                        SET times_seen = times_seen + 1,
                            last_seen_at = now()
                        WHERE food_key = %s
                    """, (food_key,))

            conn.commit()

            # 6. If there are misses, output batch for agent to estimate
            if misses:
                # ── SELF-HEALING (2026-10-02): persist progress even when the
                # model round-trip for novel foods is slow/fails. Entries whose
                # items were ALL cache hits are marked parsed now, and their
                # dates are rebuilt, so a timeout never strands the whole batch
                # and the next run only sees the genuinely novel entries.
                miss_entry_ids = {m["entry_id"] for m in misses}
                hit_only_entries = [e for e in entries if e["id"] not in miss_entry_ids]
                hit_only_dates = {e["consumed_date"] for e in hit_only_entries}
                if hit_only_entries:
                    with conn.cursor() as wcur:
                        for entry in hit_only_entries:
                            wcur.execute("""
                                UPDATE "{SCHEMA}"."food_log_entry"
                                SET parse_status = 'parsed',
                                    parse_attempts = parse_attempts + 1
                                WHERE id = %s
                            """, (entry["id"],))
                    for d in hit_only_dates:
                        rebuild_day(conn, d)
                    conn.commit()

                batch_output = {
                    "status": "needs_estimation",
                    "entries": len(entries),
                    "hits": len(item_writes),
                    "misses": len(misses),
                    "parsed_hit_only": len(hit_only_entries),
                    "batch_input": "\n".join(
                        f"{m['id']}|{m['food_key']}|{m['qty']}|{m['unit']}"
                        for m in misses
                    ),
                    "miss_items": misses,
                    "affected_dates": [d.isoformat() for d in affected_dates],
                }
                print(json.dumps(batch_output, indent=2))
                sys.exit(2)  # signal: needs model
            else:
                # All hits, mark entries parsed
                with conn.cursor() as wcur:
                    for entry in entries:
                        wcur.execute("""
                            UPDATE "{SCHEMA}"."food_log_entry"
                            SET parse_status = 'parsed',
                                parse_attempts = parse_attempts + 1
                            WHERE id = %s
                        """, (entry["id"],))
                conn.commit()

                # Rebuild affected days
                for d in affected_dates:
                    rebuild_day(conn, d)
                conn.commit()

                result = {"status": "complete", "hits": len(item_writes), "misses": 0}
                print(json.dumps(result))

    finally:
        conn.close()


def _run_batch_for_date(conn, batch_date: date) -> dict:
    """Run the full batch pipeline for one date. Assumes cache is populated.
    Used by --affected-dates auto-complete after --write-cache fills cache
    for novel foods, so the whole pipeline completes atomically in one call."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, raw_text, consumed_date
            FROM "{SCHEMA}"."food_log_entry"
            WHERE parse_status = 'pending'
              AND parse_attempts < 3
              AND superseded_by IS NULL
              AND consumed_date = %s
            ORDER BY id
        """, (batch_date,))
        entries = cur.fetchall()

        if not entries:
            return {"date": batch_date.isoformat(), "entries": 0, "status": "no_pending"}

        all_items = []
        all_food_keys = set()
        for entry in entries:
            for seq, part in enumerate(split_items(entry["raw_text"]), 1):
                qty, unit, food_key = extract_qty_unit(part)
                all_items.append((entry["id"], seq, food_key, part.strip(), qty, unit))
                all_food_keys.add(food_key)

        # Cache hits (cache was just populated by --write-cache or already exists)
        cache_hits = {}
        if all_food_keys:
            cur.execute("""
                SELECT food_key, canonical_name, unit, kcal_per_unit, confidence
                FROM "{SCHEMA}"."food_estimate_cache"
                WHERE food_key = ANY(%s)
            """, (list(all_food_keys),))
            cache_hits = {row["food_key"]: dict(row) for row in cur.fetchall()}

        item_writes = []
        remaining_misses = []
        for entry_id, seq, food_key, desc, qty, unit in all_items:
            if food_key in cache_hits:
                hit = cache_hits[food_key]
                if not units_compatible(hit["unit"], unit):
                    remaining_misses.append({"food_key": food_key, "entry_id": entry_id})
                    continue
                kcal = round(float(hit["kcal_per_unit"]) * float(qty))
                kj = round(kcal * 4.184)
                item_writes.append((entry_id, seq, food_key, desc, qty, unit,
                                   kcal, kj, hit["confidence"], 'cache'))
            else:
                remaining_misses.append({"food_key": food_key, "entry_id": entry_id})

        with conn.cursor() as wcur:
            for item in item_writes:
                wcur.execute("""
                    INSERT INTO "{SCHEMA}"."food_log_item"
                        (entry_id, seq, food_key, description, qty, unit,
                         kcal, kj, confidence, estimate_source, estimated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (entry_id, seq) DO UPDATE SET
                        food_key = EXCLUDED.food_key, kcal = EXCLUDED.kcal, kj = EXCLUDED.kj,
                        confidence = EXCLUDED.confidence, estimate_source = EXCLUDED.estimate_source
                """, item)

            for food_key in {i[2] for i in item_writes}:
                wcur.execute("""
                    UPDATE "{SCHEMA}"."food_estimate_cache"
                    SET times_seen = times_seen + 1, last_seen_at = now()
                    WHERE food_key = %s
                """, (food_key,))

            miss_ids = {m["entry_id"] for m in remaining_misses}
            for entry in entries:
                if entry["id"] not in miss_ids:
                    wcur.execute("""
                        UPDATE "{SCHEMA}"."food_log_entry"
                        SET parse_status = 'parsed', parse_attempts = parse_attempts + 1
                        WHERE id = %s
                    """, (entry["id"],))

        conn.commit()
        rebuild_day(conn, batch_date)
        conn.commit()

        return {
            "date": batch_date.isoformat(),
            "entries": len(entries),
            "hits": len(item_writes),
            "misses": len(remaining_misses),
            "status": "complete" if not remaining_misses else "partial",
        }


def rebuild_day(conn, day: date):
    """Rebuild food_day for a given date."""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM "{SCHEMA}"."food_day" WHERE consumed_date = %s
        """, (day,))

        cur.execute("""
            WITH occ AS (
                SELECT id,
                       consumed_at - LAG(consumed_at) OVER (ORDER BY consumed_at) AS gap
                FROM "{SCHEMA}"."food_log_entry"
                WHERE consumed_date = %s AND superseded_by IS NULL
            )
            INSERT INTO "{SCHEMA}"."food_day" (
                consumed_date, kcal_total, kj_total, item_count, entry_count,
                first_consumed_at, last_consumed_at, eating_window_mins,
                occasion_count, unestimated_items, rebuilt_at
            )
            SELECT
                e.consumed_date,
                SUM(i.kcal)::integer AS kcal_total,
                SUM(i.kj)::integer AS kj_total,
                COUNT(i.id)::smallint AS item_count,
                COUNT(DISTINCT e.id)::smallint AS entry_count,
                MIN(e.consumed_at) AS first_consumed_at,
                MAX(e.consumed_at) AS last_consumed_at,
                CASE
                    WHEN MIN(e.consumed_at) IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (MAX(e.consumed_at) - MIN(e.consumed_at)))::integer / 60
                END AS eating_window_mins,
                (SELECT COUNT(*)::smallint FROM occ WHERE occ.gap IS NULL OR occ.gap > interval '90 minutes')
                AS occasion_count,
                COUNT(i.id) FILTER (WHERE i.kcal IS NULL)::smallint AS unestimated_items,
                now()
            FROM "{SCHEMA}"."food_log_entry" e
            LEFT JOIN "{SCHEMA}"."food_log_item" i ON i.entry_id = e.id
            WHERE e.consumed_date = %s
              AND e.superseded_by IS NULL
            GROUP BY e.consumed_date
        """, (day, day))


if __name__ == "__main__":
    main()
