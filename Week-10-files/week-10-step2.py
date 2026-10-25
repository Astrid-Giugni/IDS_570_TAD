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

sample_text = merchant_records[0]["text"]

doc = nlp(sample_text)

print("Processed one document.")
print("Number of tokens:", len(doc))

for ent in doc.ents:
    print(ent.text, "|", ent.label_)

from collections import Counter

Counter([ent.label_ for ent in doc.ents]) 

entities_one_doc = []

for ent in doc.ents:
    entities_one_doc.append({
        "doc_id": merchant_records[0]["doc_id"],
        "entity_text": ent.text,
        "entity_label": ent.label_
    })

print("Extracted entities:", len(entities_one_doc))
entities_one_doc[:5]

all_entities = []

CHUNK_SIZE = 50000  # characters per chunk

for record in merchant_records:
    text = record["text"]
    
    # Split text into chunks
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        doc = nlp(chunk)
        
        for ent in doc.ents:
            all_entities.append({
                "doc_id": record["doc_id"],
                "entity_text": ent.text,
                "entity_label": ent.label_
            })

print("Total entities extracted:", len(all_entities))

import json
from pathlib import Path

OUT_ENTS_RAW = Path.cwd() / "merchant_entities_raw.json"

with open(OUT_ENTS_RAW, "w", encoding="utf-8") as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

print("Saved raw entity mentions:", OUT_ENTS_RAW.resolve())