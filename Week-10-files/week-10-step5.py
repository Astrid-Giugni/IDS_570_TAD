## Load merchant texts + tuned Spacy Model
import json
import re
from pathlib import Path
from collections import Counter
import spacy

# Inputs
MERCHANT_PATH = Path.cwd() / "merchant_texts_for_spacy.json"
TUNED_MODEL_DIR = Path.cwd() / "models" / "spacy_loc_tuned"

# Load merchant records
with open(MERCHANT_PATH, "r", encoding="utf-8") as f:
    merchant_records = json.load(f)

print("Loaded merchant texts:", len(merchant_records))

# Load tuned model
nlp = spacy.load(TUNED_MODEL_DIR)
print("Loaded tuned spaCy model:", TUNED_MODEL_DIR.resolve())

CHUNK_SIZE = 50000
KEEP_LABELS = {"GPE", "LOC"}

def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

locations = []   # mention-level table

for record in merchant_records:
    doc_id = record["doc_id"]
    text = record["text"]

    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        doc = nlp(chunk)

        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                locations.append({
                    "doc_id": doc_id,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    # offsets are within the chunk, not the full document:
                    "start_char_in_chunk": ent.start_char,
                    "end_char_in_chunk": ent.end_char,
                })

print("Total location mentions extracted:", len(locations))

## Chunked extraction + keep doc_id

CHUNK_SIZE = 50000
KEEP_LABELS = {"GPE", "LOC"}

def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

locations = []   # mention-level table

for record in merchant_records:
    doc_id = record["doc_id"]
    text = record["text"]

    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        doc = nlp(chunk)

        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                locations.append({
                    "doc_id": doc_id,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    # offsets are within the chunk, not the full document:
                    "start_char_in_chunk": ent.start_char,
                    "end_char_in_chunk": ent.end_char,
                })

print("Total location mentions extracted:", len(locations))

#Save the mention level output (so we don't have to re-run this again)

OUT_MENTIONS = Path.cwd() / "merchant_locations_mentions_tuned.json"
with open(OUT_MENTIONS, "w", encoding="utf-8") as f:
    json.dump(locations, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_MENTIONS.resolve())

# Create counts for dedupe/entity resolution

counts_by_label = {
    "GPE": Counter(),
    "LOC": Counter()
}

for m in locations:
    norm = normalize_ent(m["entity_text"])
    counts_by_label[m["entity_label"]][norm] += 1

print("\nTop 20 GPE (tuned):")
for name, n in counts_by_label["GPE"].most_common(20):
    print(f"{n:>7}  {name}")

print("\nTop 20 LOC (tuned):")
for name, n in counts_by_label["LOC"].most_common(20):
    print(f"{n:>7}  {name}")
    
# save counts

OUT_COUNTS = Path.cwd() / "merchant_locations_counts_tuned.json"
counts_out = {lab: dict(cnt.most_common()) for lab, cnt in counts_by_label.items()}

with open(OUT_COUNTS, "w", encoding="utf-8") as f:
    json.dump(counts_out, f, ensure_ascii=False, indent=2)

print("\nSaved counts:", OUT_COUNTS.resolve())   

# Compare with base model:

nlp_base = spacy.load("en_core_web_sm")

def extract_counts(nlp_model):
    locs = []
    for record in merchant_records:
        t = record["text"]
        for i in range(0, len(t), CHUNK_SIZE):
            doc = nlp_model(t[i:i+CHUNK_SIZE])
            for ent in doc.ents:
                if ent.label_ in KEEP_LABELS:
                    locs.append(normalize_ent(ent.text))
    return Counter(locs)

base_counts = extract_counts(nlp_base)
tuned_counts = Counter()
for lab in counts_by_label:
    tuned_counts.update(counts_by_label[lab])

print("\nTOTAL locations (base):", sum(base_counts.values()))
print("TOTAL locations (tuned):", sum(tuned_counts.values()))

print("\nTop 15 base locations:")
for name, n in base_counts.most_common(15):
    print(f"{n:>7}  {name}")

print("\nTop 15 tuned locations:")
for name, n in tuned_counts.most_common(15):
    print(f"{n:>7}  {name}")