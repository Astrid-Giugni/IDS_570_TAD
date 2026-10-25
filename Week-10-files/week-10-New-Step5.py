"""
extract_merchant_locations.py
=============================
Applies the fine-tuned spaCy NER model to a collection of merchant texts,
extracts all GPE and LOC entities, and saves both mention-level results
and aggregated frequency counts.

Input:  merchant_texts_for_spacy.json
        — a JSON list of {"doc_id": ..., "text": ...} records

        models/spacy_loc_tuned/
        — the fine-tuned spaCy model from train_spacy_ner.py

Output: merchant_locations_mentions_tuned.json
        — every individual entity mention with doc_id, text, label, and
          character offsets into the full document

        merchant_locations_counts_tuned.json
        — normalised frequency counts grouped by label (GPE / LOC)

Usage:  python extract_merchant_locations.py
"""

# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import json
import re
from pathlib import Path
from collections import Counter

import spacy

# ──────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
MERCHANT_PATH   = Path.cwd() / "merchant_texts_for_spacy.json"
TUNED_MODEL_DIR = Path.cwd() / "models" / "spacy_loc_tuned"
OUT_MENTIONS    = Path.cwd() / "merchant_locations_mentions_tuned.json"
OUT_COUNTS      = Path.cwd() / "merchant_locations_counts_tuned.json"

CHUNK_SIZE  = 50_000       # max characters per chunk sent to spaCy
KEEP_LABELS = {"GPE", "LOC"}

# ──────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
def normalize_ent(text: str) -> str:
    """
    Normalise an entity's surface text for counting and deduplication.
    Strips whitespace, collapses internal runs of whitespace, and
    lowercases so that 'Virginia', ' virginia', and 'VIRGINIA' all
    map to the same key.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def find_chunk_end(text: str, start: int, max_size: int) -> int:
    """
    Find a safe chunk boundary that doesn't split a word.

    Starting from `start`, we look up to `max_size` characters ahead.
    If that landing point is mid-word, we back up to the nearest
    whitespace so no token is cut in half.  This prevents the NER
    model from seeing partial words at chunk edges, which could cause
    it to miss an entity or hallucinate one.
    """
    end = min(start + max_size, len(text))

    # If we've reached the end of the text, no adjustment needed
    if end >= len(text):
        return end

    # If we landed on whitespace, we're fine
    if text[end].isspace():
        return end

    # Otherwise back up to the nearest space within this chunk
    safe_end = text.rfind(" ", start, end)

    # If there's no space at all in this chunk (one giant token?),
    # fall back to the hard boundary rather than looping forever
    if safe_end <= start:
        return end

    return safe_end

# ──────────────────────────────────────────────────────────────────────
# 4. LOAD DATA AND MODEL
# ──────────────────────────────────────────────────────────────────────
with open(MERCHANT_PATH, "r", encoding="utf-8") as f:
    merchant_records = json.load(f)
print(f"Loaded merchant texts: {len(merchant_records)}")

nlp = spacy.load(TUNED_MODEL_DIR)
print(f"Loaded tuned spaCy model: {TUNED_MODEL_DIR.resolve()}")

# ──────────────────────────────────────────────────────────────────────
# 5. EXTRACT ENTITIES
#
#    For each document we split the text into chunks (most documents
#    will fit in a single chunk) and run the NER pipeline on each.
#    Character offsets are recorded relative to the FULL document,
#    not the chunk, so they can be used later to highlight entities
#    in the original text.
# ──────────────────────────────────────────────────────────────────────
locations = []

for record in merchant_records:
    doc_id = record["doc_id"]
    text   = record["text"]

    # Walk through the text in chunks
    chunk_start = 0
    while chunk_start < len(text):
        chunk_end = find_chunk_end(text, chunk_start, CHUNK_SIZE)
        chunk     = text[chunk_start:chunk_end]

        doc = nlp(chunk)

        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                locations.append({
                    "doc_id":       doc_id,
                    "entity_text":  ent.text,
                    "entity_label": ent.label_,
                    # Offsets into the full document text
                    "start_char":   chunk_start + ent.start_char,
                    "end_char":     chunk_start + ent.end_char,
                })

        chunk_start = chunk_end

print(f"Total location mentions extracted: {len(locations)}")

# ──────────────────────────────────────────────────────────────────────
# 6. SAVE MENTION-LEVEL OUTPUT
#    One record per entity mention — useful for downstream analysis,
#    joining back to document metadata, or reviewing model output.
# ──────────────────────────────────────────────────────────────────────
with open(OUT_MENTIONS, "w", encoding="utf-8") as f:
    json.dump(locations, f, ensure_ascii=False, indent=2)
print(f"Saved mentions: {OUT_MENTIONS.resolve()}")

# ──────────────────────────────────────────────────────────────────────
# 7. AGGREGATE COUNTS
#    Normalise each entity's text and count occurrences by label.
#    This gives you a quick overview of what the model found and
#    which locations appear most frequently across the corpus.
# ──────────────────────────────────────────────────────────────────────
counts_by_label = {label: Counter() for label in KEEP_LABELS}

for m in locations:
    norm = normalize_ent(m["entity_text"])
    counts_by_label[m["entity_label"]][norm] += 1

print("\nTop 20 GPE (tuned):")
for name, n in counts_by_label["GPE"].most_common(20):
    print(f"  {n:>5}  {name}")

print("\nTop 20 LOC (tuned):")
for name, n in counts_by_label["LOC"].most_common(20):
    print(f"  {n:>5}  {name}")

# ──────────────────────────────────────────────────────────────────────
# 8. SAVE COUNTS
# ──────────────────────────────────────────────────────────────────────
counts_out = {
    label: dict(counter.most_common())
    for label, counter in counts_by_label.items()
}

with open(OUT_COUNTS, "w", encoding="utf-8") as f:
    json.dump(counts_out, f, ensure_ascii=False, indent=2)
print(f"\nSaved counts: {OUT_COUNTS.resolve()}")