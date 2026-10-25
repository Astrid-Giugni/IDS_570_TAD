import json
import re
from pathlib import Path
from collections import Counter
import spacy

MERCHANT_PATH = Path.cwd() / "merchant_texts_for_spacy.json"
OUT_COUNTS = Path.cwd() / "merchant_locations_counts_base.json"

with open(MERCHANT_PATH, "r", encoding="utf-8") as f:
    merchant_records = json.load(f)

nlp = spacy.load("en_core_web_sm")

CHUNK_SIZE = 50000
KEEP_LABELS = {"GPE", "LOC"}

def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

counts = Counter()

for record in merchant_records:
    text = record["text"]
    for i in range(0, len(text), CHUNK_SIZE):
        doc = nlp(text[i:i+CHUNK_SIZE])
        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                counts[normalize(ent.text)] += 1

with open(OUT_COUNTS, "w", encoding="utf-8") as f:
    json.dump(dict(counts), f, ensure_ascii=False, indent=2)

print("Saved base counts to:", OUT_COUNTS.resolve())
