import json
from pathlib import Path

BASE_COUNTS_PATH  = Path.cwd() / "merchant_locations_counts_base.json"
TUNED_COUNTS_PATH = Path.cwd() / "merchant_locations_counts_tuned.json"

with open(BASE_COUNTS_PATH, "r", encoding="utf-8") as f:
    base_raw = json.load(f)

with open(TUNED_COUNTS_PATH, "r", encoding="utf-8") as f:
    tuned_raw = json.load(f)

# Both files are nested: {"GPE": {...}, "LOC": {...}}
# Merge GPE + LOC into one flat dict for each

def merge_labels(data):
    merged = {}
    for label in data:
        for k, v in data[label].items():
            merged[k] = merged.get(k, 0) + v
    return merged

base_counts  = merge_labels(base_raw)
tuned_counts = merge_labels(tuned_raw)

all_places = set(base_counts) | set(tuned_counts)

diff = {
    place: tuned_counts.get(place, 0) - base_counts.get(place, 0)
    for place in all_places
}

print("\nBiggest increases:")
for place, delta in sorted(diff.items(), key=lambda x: -x[1])[:15]:
    print(f"{delta:>6}  {place}")

print("\nBiggest decreases:")
for place, delta in sorted(diff.items(), key=lambda x: x[1])[:15]:
    print(f"{delta:>6}  {place}")