#!/usr/bin/env python3
"""
food_ingest.py — Parse and store a food log entry from raw text.

Usage:
  python3 food_ingest.py "<raw_text>" [--source-msg-id <id>] [--input-mode text|voice]
                                    [--consumed-at <ISO-timestamptz>]

Output: JSON with the inserted entry id and parsed fields.

The caller (agent) pre-strips the "log" keyword before invoking this script.
The agent also handles +/- planned/unplanned parsing and passes the cleaned text.

Steps:
  1. Strip leading +/- markers, extract planned boolean
  2. Normalize number-words ("to"→"two", "ate"→"eight", etc.)
  3. Resolve consumed_at timestamp
  4. Infer occasion_type (meal/snack/drink)
  5. INSERT into food_log_entry
  6. Return entry id + summary
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime, time, timezone, timedelta
from typing import Optional, Tuple
from collections import OrderedDict
from zoneinfo import ZoneInfo

import psycopg2

SCHEMA = os.environ.get("PGSCHEMA", "DW Health")

DB = os.environ.get("PGDATABASE", "postgres")
TZ = ZoneInfo("America/Vancouver")  # DST-aware

# ── NUMBER WORD NORMALIZATION ──────────────────────────────────────────

# Map spoken number-words that Whisper commonly mishears for food quantities
NUMBER_WORDS = {
    # direct numbers
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    # common voice transcription errors
    "to": 2,       # "to slices of toast" → 2
    "too": 2,      # "too slices"
    "ate": 8,      # "ate cookies" could be "eight cookies"
    "for": 4,      # "for slices"
    "fore": 4,     # "fore slices"
    # fractions
    "half": 0.5,
    "quarter": 0.25,
    "third": 0.33,
    "couple": 2,
    "dozen": 12,
}

def normalize_number_words(text: str) -> str:
    """Replace word-numbers *only* when they act as quantities (next word is
    a unit or food word). Verbs/prepositions like "ate", "to", "for" stay
    intact when followed by ordinary words."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        stripped = w.strip(",.;:!?")
        # Look at the next significant word
        nxt = words[i + 1].strip(",.;:!?").lower() if i + 1 < len(words) else None
        # Only treat as quantity if next word is a unit or a food-like word
        if stripped.lower() in NUMBER_WORDS and nxt and (nxt in UNIT_WORDS or looks_like_food(nxt)):
            val = NUMBER_WORDS[stripped.lower()]
            if val == int(val):
                result.append(str(int(val)))
            else:
                result.append(str(val))
        else:
            result.append(words[i])
        i += 1
    return " ".join(result)

# Words that strongly suggest a food unit follows a number
UNIT_WORDS = {
    "slices", "slice", "pieces", "piece", "cups", "cup", "bowls", "bowl",
    "servings", "serving", "handfuls", "handful", "glasses", "glass",
    "cans", "can", "bottles", "bottle", "bars", "bar", "scoops", "scoop",
    "plates", "plate", "bowls", "bowl", "eggs", "egg", "cookies", "cookie",
    "bites", "bite", "rolls", "roll", "mugs", "mug",
}

# Common food nouns (used to decide whether a number-word is a quantity)
FOOD_WORDS = {
    "toast", "slices", "sandwich", "sandwiches", "cookies", "cookie",
    "pizza", "slices", "eggs", "egg", "bagel", "bagels", "muffin",
    "muffins", "banana", "bananas", "apples", "apple", "oranges",
    "orange", "chicken", "burger", "burgers", "tacos", "taco", "sushi",
    "rolls", "rolls", "pieces", "piece", 
}

def looks_like_food(word: str) -> bool:
    """Heuristic: a word that ends in a common food-sounding suffix or is a
    known food word. Cheap and case-insensitive."""
    w = word.lower().rstrip(".s")
    return w in FOOD_WORDS or w.endswith(("ies", "oes"))

# ── FOOD KEY NORMALIZATION ────────────────────────────────────────────

STOP_WORDS = {"a", "an", "some", "of", "with", "my", "the"}

def normalize_food_key(text: str) -> str:
    """
    Normalize raw food text to food_key.
    lowercase → strip punctuation → collapse whitespace → drop stopwords.
    Does NOT stem.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation → space
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    # Drop leading quantity words
    while words and words[0] in NUMBER_WORDS:
        words.pop(0)
    return " ".join(words) if words else text

# ── PLANNED / UNPLANNED DETECTION ─────────────────────────────────────

PLANNED_MARKER = re.compile(r'^\s*([+-])\s*')

def extract_planned(text: str) -> Tuple[Optional[bool], str]:
    """Extract +/- marker, return (planned, cleaned_text)."""
    m = PLANNED_MARKER.match(text)
    if m:
        marker = m.group(1)
        cleaned = text[m.end():].strip()
        return (True if marker == '+' else False), cleaned
    # Also check for voice keyword
    lowered = text.lower().strip()
    if lowered.startswith("planned "):
        return True, text[len("planned "):].strip()
    if lowered.startswith("unplanned "):
        return False, text[len("unplanned "):].strip()
    return None, text

# ── TIMESTAMP INFERENCE ──────────────────────────────────────────────

# Regex patterns for explicit times in text
TIME_PATTERNS = [
    (re.compile(r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)', re.I), 'explicit'),
    (re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', re.I), 'explicit'),
    (re.compile(r'(?:at\s+)?(\d{1,2}):(\d{2})(?!\s*[ap]m)', re.I), 'explicit'),
    (re.compile(r'(?:at\s+)?noon', re.I), 'explicit'),
    (re.compile(r'(?:at\s+)?midnight', re.I), 'explicit'),
]

RELATIVE_PATTERNS = [
    (re.compile(r'(?:just\s+)?now', re.I), timedelta(minutes=0)),
    (re.compile(r'(\d+)\s*min(?:ute)?s?\s*(?:ago)?', re.I), None),  # dynamic
    (re.compile(r'an?\s*hour\s*(?:ago)?', re.I), timedelta(hours=1)),
    (re.compile(r'(\d+)\s*hours?\s*(?:ago)?', re.I), None),
    (re.compile(r'this\s+morning', re.I), None),  # resolved below
    (re.compile(r'this\s+afternoon', re.I), None),
    (re.compile(r'this\s+evening', re.I), None),
    (re.compile(r'last\s+night', re.I), None),
    (re.compile(r'yesterday', re.I), timedelta(days=1)),
]

def resolve_consumed_at(text: str, recorded_at: datetime) -> tuple[datetime, str]:
    """Return (consumed_at, time_source)."""
    # Try explicit time extraction
    for pattern, method in TIME_PATTERNS:
        m = pattern.search(text)
        if m:
            pat_str = pattern.pattern
            if 'noon' in pat_str:
                t = time(12, 0)
            elif 'midnight' in pat_str:
                t = time(0, 0)
            else:
                hour = int(m.group(1))
                minute = int(m.group(2) or 0)
                ampm = m.group(3).lower() if m.lastindex and m.lastindex >= 3 and m.group(3) else None
                if ampm:
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    if ampm == 'am' and hour == 12:
                        hour = 0
                else:
                    # Bare time — infer am/pm
                    if hour <= 8:
                        hour += 0  # assume AM for early times
                    elif hour <= 12:
                        pass  # ambiguous, leave as-is
                t = time(hour, minute)
            # Use today's date from recorded_at
            local_dt = recorded_at.astimezone(TZ)
            consumed = datetime(local_dt.year, local_dt.month, local_dt.day, t.hour, t.minute, 0, 0, tzinfo=TZ)
            # If it's more than 18h before recorded_at, maybe it's yesterday
            if (recorded_at - consumed).total_seconds() > 18 * 3600:
                consumed += timedelta(days=1)
            return consumed.astimezone(timezone.utc), 'explicit'

    # Try relative phrases
    for pattern, delta in RELATIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            if pattern.pattern == '(?:just\\s+)?now':
                return recorded_at, 'message'
            elif 'this morning' in (m.group(0)):
                t = time(8, 0)
                local_dt = recorded_at.astimezone(TZ)
                consumed = datetime(local_dt.year, local_dt.month, local_dt.day, t.hour, t.minute, tzinfo=TZ)
                return consumed.astimezone(timezone.utc), 'message'
            elif 'this afternoon' in (m.group(0)):
                t = time(13, 0)
                local_dt = recorded_at.astimezone(TZ)
                consumed = datetime(local_dt.year, local_dt.month, local_dt.day, t.hour, t.minute, tzinfo=TZ)
                return consumed.astimezone(timezone.utc), 'message'
            elif 'this evening' in (m.group(0)):
                t = time(19, 0)
                local_dt = recorded_at.astimezone(TZ)
                consumed = datetime(local_dt.year, local_dt.month, local_dt.day, t.hour, t.minute, tzinfo=TZ)
                return consumed.astimezone(timezone.utc), 'message'
            elif 'last night' in (m.group(0)):
                t = time(20, 0)
                local_dt = recorded_at.astimezone(TZ) - timedelta(days=1)
                consumed = datetime(local_dt.year, local_dt.month, local_dt.day, t.hour, t.minute, tzinfo=TZ)
                return consumed.astimezone(timezone.utc), 'message'
            elif delta:
                return (recorded_at - delta).astimezone(timezone.utc), 'message'
            elif m.lastindex:
                # Dynamic: e.g. "30 minutes ago", "2 hours ago"
                val = int(m.group(1))
                if "min" in m.group(0):
                    delta = timedelta(minutes=val)
                elif "hour" in m.group(0):
                    delta = timedelta(hours=val)
                return (recorded_at - delta).astimezone(timezone.utc), 'message'

    # Fallback
    return recorded_at.astimezone(timezone.utc), 'message'

# ── OCCASION TYPE INFERENCE ───────────────────────────────────────────

def infer_occasion_type(consumed_at: datetime, item_count: int,
                         raw_text: str) -> str:
    """
    Infer meal/snack/drink from time of day and content.
    Heuristic, cheap, no model.
    """
    local = consumed_at.astimezone(TZ)
    hour = local.hour

    # Detecting drink-only entries
    drink_only_keywords = {"coffee", "tea", "latte", "cappuccino", "flat white",
                           "americano", "espresso", "smoothie", "milk", "juice",
                           "water", "beer", "wine", "cocktail", "soda", "pop"}
    text_lower = raw_text.lower()
    if any(kw in text_lower for kw in drink_only_keywords) and item_count == 1:
        if not any(food_kw in text_lower for food_kw in
                   {"toast", "eggs", "sandwich", "bagel", "croissant",
                    "pastry", "muffin", "cake", "cookie", "biscuit",
                    "cereal", "porridge", "oatmeal", "granola", "yogurt",
                    "fruit", "banana", "apple", "salad", "soup", "rice",
                    "pasta", "chicken", "meat", "fish", "burger", "pizza"}):
            return 'drink'

    # Meal vs snack by time + item count
    # Traditional meal times
    # Single items that sound like full meals (high calorie, bowl/plate) = meal
    meal_keywords = {'breakfast', 'lunch', 'dinner', 'supper', 'meal',
                     'bowl', 'plate', 'platter', 'burger', 'pizza',
                     'pasta', 'curry', 'stir fry', 'stew', 'roast',
                     'porridge', 'oatmeal', 'cereal', 'granola'}
    sounds_like_meal = any(kw in text_lower for kw in meal_keywords)

    if 6 <= hour <= 10:
        if item_count >= 2 or sounds_like_meal:
            return 'meal'
        else:
            return 'snack'
    elif 11 <= hour <= 13:
        return 'meal'  # any size at lunchtime
    elif 14 <= hour <= 16:
        return 'snack'
    elif 17 <= hour <= 20:
        if item_count >= 2 or sounds_like_meal:
            return 'meal'
        else:
            return 'snack'
    else:  # late night / very early
        return 'snack' if item_count <= 2 else 'meal'

# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest a food log entry")
    parser.add_argument("raw_text", help="Cleaned food text (log prefix already stripped)")
    parser.add_argument("--source-msg-id", default=None)
    parser.add_argument("--input-mode", default="text", choices=["text", "voice"])
    parser.add_argument("--consumed-at", default=None,
                        help="ISO timestamp override for consumed_at")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse but don't write to DB")
    args = parser.parse_args()

    raw = args.raw_text.strip()
    if not raw:
        print(json.dumps({"error": "empty text"}))
        sys.exit(1)

    recorded_at = datetime.now(timezone.utc)

    # 1. Extract planned/unplanned marker
    planned, cleaned = extract_planned(raw)

    # 2. Normalize number-words
    cleaned = normalize_number_words(cleaned)

    # 3. Resolve consumed_at
    if args.consumed_at:
        consumed_at = datetime.fromisoformat(args.consumed_at)
        if consumed_at.tzinfo is None:
            consumed_at = consumed_at.replace(tzinfo=timezone.utc)
        time_source = 'corrected'
    else:
        consumed_at, time_source = resolve_consumed_at(cleaned, recorded_at)

    # consumed_date in America/Vancouver
    consumed_date = consumed_at.astimezone(TZ).date()

    # 4. Rough item count for occasion inference (regex split)
    # Split on commas, "and", "&", "plus"
    raw_for_count = re.sub(r'\b(and|&|plus|with)\b', ',', cleaned, flags=re.I)
    parts = [p.strip() for p in raw_for_count.split(',') if p.strip()]
    # Don't count drink keywords as separate items if they're obviously modifiers
    rough_count = len(parts) if parts else 1

    # 5. Infer occasion_type
    occasion_type = infer_occasion_type(consumed_at, rough_count, cleaned)

    if args.dry_run:
        result = {
            "raw_text": raw,
            "cleaned_text": cleaned,
            "planned": planned,
            "consumed_at": consumed_at.isoformat(),
            "consumed_date": consumed_date.isoformat(),
            "time_source": time_source,
            "occasion_type": occasion_type,
            "input_mode": args.input_mode,
            "rough_items": parts,
        }
        print(json.dumps(result, indent=2))
        return

    # 6. Write to DB
    # ⚠️  Check for duplicate source_msg_id
    if args.source_msg_id:
        existing = psycopg2.connect(f"dbname={DB}")
        try:
            with existing.cursor() as cur:
                cur.execute(
                    f'SELECT id FROM "{SCHEMA}"."food_log_entry" WHERE source_msg_id = %s',
                    (args.source_msg_id,))
                row = cur.fetchone()
                if row:
                    print(json.dumps({"duplicate": True, "existing_id": row[0]}))
                    return
        finally:
            existing.close()

    conn = psycopg2.connect(f"dbname={DB}")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO "{SCHEMA}"."food_log_entry"
                    (recorded_at, consumed_at, consumed_date, raw_text,
                     source_msg_id, input_mode, time_source, parse_status,
                     planned, occasion_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
                RETURNING id
            """, (recorded_at, consumed_at, consumed_date, cleaned,
                  args.source_msg_id, args.input_mode, time_source,
                  planned, occasion_type))
            entry_id = cur.fetchone()[0]
        conn.commit()

        result = {
            "entry_id": entry_id,
            "cleaned_text": cleaned,
            "planned": planned,
            "consumed_at": consumed_at.isoformat(),
            "consumed_date": consumed_date.isoformat(),
            "time_source": time_source,
            "occasion_type": occasion_type,
            "input_mode": args.input_mode,
        }
        print(json.dumps(result))

    finally:
        conn.close()

if __name__ == "__main__":
    main()