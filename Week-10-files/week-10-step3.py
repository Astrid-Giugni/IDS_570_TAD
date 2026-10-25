# 1. Imports
import json
import re
from pathlib import Path
from collections import Counter

# 2. Load entity mentions
ENT_PATH = Path.cwd() / "merchant_entities_raw.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    all_entities = json.load(f)

print("Loaded entity mentions:", len(all_entities))

# 3. Normalization
def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# 4. Count
entity_text_counts = Counter(
    normalize_ent(e["entity_text"]) for e in all_entities
)

print("Unique entity strings:", len(entity_text_counts))
print("Top 20 entity strings:")
for ent_text, n in entity_text_counts.most_common(20):
    print(f"{n:>7}  {ent_text}")

from collections import Counter

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC"}

filtered = [
    e for e in all_entities
    if e["entity_label"] in KEEP_LABELS
]

print("Filtered entity mentions:", len(filtered))

entity_text_counts_filtered = Counter(
    normalize_ent(e["entity_text"]) for e in filtered
)


#### Count entity strings for desired labels ONLY
from collections import Counter

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

filtered = [
    e for e in all_entities
    if e["entity_label"] in KEEP_LABELS
]

print("Filtered entity mentions:", len(filtered))

entity_text_counts_filtered = Counter(
    normalize_ent(e["entity_text"]) for e in filtered
)

print("Unique entity strings (filtered):", len(entity_text_counts_filtered))
print("Top 20 entity strings (filtered):")
for ent_text, n in entity_text_counts_filtered.most_common(20):
    print(f"{n:>7}  {ent_text}")

## Count by label:

counts_by_label = {lab: Counter() for lab in KEEP_LABELS}

for e in filtered:
    lab = e["entity_label"]
    txt = normalize_ent(e["entity_text"])
    counts_by_label[lab][txt] += 1

for lab in ["ORG", "GPE", "PERSON", "LOC", "NORP"]:
    print(f"\nTop 15 {lab}:")
    for ent_text, n in counts_by_label[lab].most_common(15):
        print(f"{n:>7}  {ent_text}")

#-----------------------------------------------------
####FOR ME ONLY: CREATE A "gold standard set using GPT"
#-----------------------------------------------------

import json, random, re
from pathlib import Path

MERCHANT_PATH = Path.cwd() / "merchant_texts_for_spacy.json"
with open(MERCHANT_PATH, "r", encoding="utf-8") as f:
    merchant_records = json.load(f)

print("Loaded merchant records:", len(merchant_records))

def pseudo_paragraphs(text: str, chunk_chars: int = 1000, min_chars: int = 600):
    text = re.sub(r"\s+", " ", text).strip()
    blocks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        block = text[start:end].strip()
        if len(block) >= min_chars:
            blocks.append(block)
        start = end
    return blocks

random.seed(42)
sample_records = random.sample(merchant_records, k=min(10, len(merchant_records)))

paragraphs = []
for r in sample_records:
    paragraphs.extend(pseudo_paragraphs(r["text"], chunk_chars=1100, min_chars=600))

print("Collected pseudo-paragraphs:", len(paragraphs))

paragraphs = paragraphs[:50]
print("Final paragraphs for annotation:", len(paragraphs))

OUT = Path.cwd() / "location_annotation_sample.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(paragraphs, f, ensure_ascii=False, indent=2)

print("Saved:", OUT.resolve())

###---------------------------------------------
### DONE
###-----------------------------------------

import json
from pathlib import Path

TRAIN_PATH = Path.cwd() / "location_train_data.json"

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    TRAIN_DATA = json.load(f)

print("Loaded training examples:", len(TRAIN_DATA))

import random
random.seed(42)

random.shuffle(TRAIN_DATA)
split = int(0.8 * len(TRAIN_DATA))
TRAIN = TRAIN_DATA[:split]
DEV   = TRAIN_DATA[split:]

print("Train:", len(TRAIN), "Dev:", len(DEV))