import json
from pathlib import Path

ANN_PATH = Path.cwd() / "week10_labelled_training.json"

with open(ANN_PATH, "r", encoding="utf-8") as f:
    ann = json.load(f)

def validate_record(rec):
    text = rec["text"]
    ents = rec.get("entities", [])
    for e in ents:
        start, end, label = e["start"], e["end"], e["label"]
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start < end <= len(text)
        assert label in {"GPE", "LOC"}
        span = text[start:end]
        assert span.strip() != "", "Empty/whitespace span"
    return True

bad = 0
for i, rec in enumerate(ann):
    try:
        validate_record(rec)
    except Exception as ex:
        bad += 1
        print(f"\nRecord {i} FAILED: {ex}")
        print("Text snippet:", rec["text"][:200])
        print("Entities:", rec.get("entities"))

print("\nTotal records:", len(ann))
print("Bad records:", bad)

TRAIN_DATA = []
for rec in ann:
    ents = [(e["start"], e["end"], e["label"]) for e in rec.get("entities", [])]
    TRAIN_DATA.append((rec["text"], {"entities": ents}))

print("Training examples:", len(TRAIN_DATA))

# After creating TRAIN_DATA in Step 1

import json
from pathlib import Path

OUT_TRAIN = Path.cwd() / "location_train_data.json"

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(TRAIN_DATA, f, ensure_ascii=False, indent=2)

print("Saved training data:", OUT_TRAIN.resolve())