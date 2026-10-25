import json
from pathlib import Path

import json
from pathlib import Path

with open(Path.cwd() / "merchant_locations_counts_base.json", "r") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys())[:5])
# Should show: ['GPE', 'LOC']
# If it shows place names like ['portugal', 'england', ...], the file is flat

BASE_COUNTS_PATH  = Path.cwd() / "merchant_locations_counts_base.json"
TUNED_COUNTS_PATH = Path.cwd() / "merchant_locations_counts_tuned.json"

with open(BASE_COUNTS_PATH, "r", encoding="utf-8") as f:
    base_counts = json.load(f)

with open(TUNED_COUNTS_PATH, "r", encoding="utf-8") as f:
    tuned_counts_raw = json.load(f)

# merge tuned GPE + LOC
tuned_counts = {}
for label in tuned_counts_raw:
    for k, v in tuned_counts_raw[label].items():
        tuned_counts[k] = tuned_counts.get(k, 0) + v


# merge base GPE + LOC (same structure as tuned)
base_merged = {}
for label in base_counts:
    for k, v in base_counts[label].items():
        base_merged[k] = base_merged.get(k, 0) + v

# merge tuned GPE + LOC
tuned_counts = {}
for label in tuned_counts_raw:
    for k, v in tuned_counts_raw[label].items():
        tuned_counts[k] = tuned_counts.get(k, 0) + v

all_places = set(base_merged) | set(tuned_counts)

diff = {
    place: tuned_counts.get(place, 0) - base_merged.get(place, 0)
    for place in all_places
}

print("\nBiggest increases:")
for place, delta in sorted(diff.items(), key=lambda x: -x[1])[:15]:
    print(f"{delta:>6}  {place}")

print("\nBiggest decreases:")
for place, delta in sorted(diff.items(), key=lambda x: x[1])[:15]:
    print(f"{delta:>6}  {place}")