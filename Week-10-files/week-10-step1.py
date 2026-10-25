import json
from pathlib import Path

# Point to the folder with our new Post Spring Break Texts (or whatever you named it)
NEW_TEXTS_DIR = Path.cwd() / "Post_Spring_Break_Texts"


# Read all .txt files
paths = sorted(NEW_TEXTS_DIR.glob("*.txt"))
print("Found .txt files:", len(paths))

# How the JSON file will be organized
records = []
for p in paths:
    text = p.read_text(encoding="utf-8", errors="ignore")
    records.append({
        "doc_id": p.stem,           # filename without extension
        "filename": p.name,         # full filename
        "text": text                # raw text
    })

# Quick check
print("First record keys:", records[0].keys())
print("First doc_id:", records[0]["doc_id"])
print("First text snippet:", records[0]["text"][:200])

# Save the corpus as JSON
OUT_JSON = Path.cwd() / "new_texts.json"

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_JSON)


from pathlib import Path
import joblib

MODEL_DIR = Path.cwd() / "models"

# Load the SAME TF-IDF vectorizer and classifier you trained in Week 08–09
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
clf        = joblib.load(MODEL_DIR / "merchant_logreg.joblib")

print("Loaded TF-IDF vectorizer + logistic regression classifier.")

texts = [r["text"] for r in records]

X_new = vectorizer.transform(texts)          # IMPORTANT: transform, not fit_transform
probs = clf.predict_proba(X_new)[:, 1]
preds = (probs >= 0.50).astype(int)

for r, p, yhat in zip(records, probs, preds):
    r["pred_prob_merchant"] = float(p)
    r["pred_merchant"] = int(yhat)

print("Classified:", len(records), "documents")
print("Predicted merchant (threshold .50):", sum(preds))

#save the classified corpus into the JSON

import json
from pathlib import Path

OUT_CLASSIFIED = Path.cwd() / "classified_texts.json"

with open(OUT_CLASSIFIED, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved classified dataset:", OUT_CLASSIFIED.resolve())

#create a merchant texts for spaCy subset with stricter threshold
THRESH = 0.70

merchant_only = [r for r in records if r["pred_prob_merchant"] >= THRESH]

OUT_MERCHANT = Path.cwd() / "merchant_texts_for_spacy.json"

with open(OUT_MERCHANT, "w", encoding="utf-8") as f:
    json.dump(merchant_only, f, ensure_ascii=False, indent=2)

print(f"High-confidence merchant texts (p >= {THRESH}):", len(merchant_only))
print("Saved:", OUT_MERCHANT.resolve())

#check top 5 most confident:
top5 = sorted(records, key=lambda r: r["pred_prob_merchant"], reverse=True)[:5]
for r in top5:
    print(r["doc_id"], r["pred_prob_merchant"])

import spacy
import json
from pathlib import Path

# Load the English model
nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded.")

MERCHANT_PATH = Path.cwd() / "merchant_texts_for_spacy.json"

with open(MERCHANT_PATH, "r", encoding="utf-8") as f:
    merchant_records = json.load(f)

print("Number of merchant texts:", len(merchant_records))