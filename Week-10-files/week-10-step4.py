"""
train_spacy_ner.py
==================
Fine-tunes spaCy's built-in NER pipeline on a small set of hand-labelled
location entities (GPE and LOC) drawn from early modern English texts.

Input:  location_train_data.json
        — a JSON list of [text, {"entities": [[start, end, label], ...]}] pairs

Output: models/spacy_loc_tuned/
        — a saved spaCy model you can reload with spacy.load("models/spacy_loc_tuned")

Usage:  python train_spacy_ner.py
"""

# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import json
import random
from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer

# ──────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
#    Adjust these values to experiment with training behaviour.
# ──────────────────────────────────────────────────────────────────────
TRAIN_PATH    = Path.cwd() / "location_train_data.json"   # path to labelled data
OUT_MODEL_DIR = Path.cwd() / "models" / "spacy_loc_tuned" # where to save the model
BASE_MODEL    = "en_core_web_sm"                           # pretrained model to start from
N_EPOCHS      = 20          # number of full passes over the training data
TRAIN_SPLIT   = 0.8         # 80 % train, 20 % dev
RANDOM_SEED   = 42          # for reproducibility
DROP_RATE     = 0.3         # dropout: randomly disables 30 % of neurons each
                            # update so the model doesn't memorise the training
                            # data too closely (helps generalisation)

# ──────────────────────────────────────────────────────────────────────
# 3. LOAD THE LABELLED DATA
#    Each item is [text_string, {"entities": [[start, end, label], ...]}]
#    which is the format spaCy's Example.from_dict expects.
# ──────────────────────────────────────────────────────────────────────
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    ALL_DATA = json.load(f)

print(f"Loaded {len(ALL_DATA)} labelled examples from {TRAIN_PATH.name}")

# ──────────────────────────────────────────────────────────────────────
# 4. TRAIN / DEV SPLIT
#    We shuffle first so the split isn't biased by document order.
#    The dev set is held out — the model never trains on it — so we
#    can use it to measure how well the model generalises.
# ──────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(ALL_DATA)

split = int(TRAIN_SPLIT * len(ALL_DATA))
TRAIN = ALL_DATA[:split]
DEV   = ALL_DATA[split:]

print(f"Train: {len(TRAIN)}   Dev: {len(DEV)}")

# ──────────────────────────────────────────────────────────────────────
# 5. LOAD THE BASE MODEL AND PREPARE THE NER PIPE
# ──────────────────────────────────────────────────────────────────────
nlp = spacy.load(BASE_MODEL)

# The NER component already exists in en_core_web_sm.  We just need to
# make sure it knows about the two labels we care about.  If they are
# already present (GPE and LOC are standard labels) this is a no-op,
# but it's good practice to be explicit.
ner = nlp.get_pipe("ner")
ner.add_label("GPE")
ner.add_label("LOC")

# ──────────────────────────────────────────────────────────────────────
# 6. CREATE THE OPTIMIZER
#
#    resume_training()  vs  initialize()
#    ─────────────────────────────────────
#    • initialize()       — resets ALL weights to random values.
#                           The model forgets everything it learned
#                           during pretraining and starts from scratch.
#                           You'd need thousands of examples to recover.
#
#    • resume_training()  — keeps the pretrained weights and returns an
#                           optimizer configured to nudge them gently.
#                           This is what "fine-tuning" actually means:
#                           the model already knows English NER and we
#                           just adapt it to our historical texts.
#
#    With only ~31 training examples, resume_training() is essential.
# ──────────────────────────────────────────────────────────────────────
optimizer = nlp.resume_training()

# ──────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
#
#    Each epoch:
#      a) Shuffle the training data (avoids the model learning order-
#         dependent patterns and smooths out loss spikes).
#      b) Loop over every example, convert it to a spaCy Example object,
#         and call nlp.update() to adjust the model weights.
#      c) Evaluate on the held-out dev set so we can track precision,
#         recall, and F1 — the metrics that actually matter for NER.
#      d) Save the model if it achieves the best dev F1 so far.
# ──────────────────────────────────────────────────────────────────────
best_f1 = 0.0               # track the best dev F1 we've seen
best_epoch = 0               # which epoch produced that best score

print(f"\n{'Epoch':>5}  {'Loss':>10}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
print("─" * 48)

for epoch in range(N_EPOCHS):

    # (a) Shuffle training data each epoch
    random.shuffle(TRAIN)

    # (b) Train on every example
    losses = {}
    for text, annots in TRAIN:
        # make_doc creates a blank Doc from the raw text (no predictions yet)
        doc = nlp.make_doc(text)
        # Example pairs a blank doc with gold-standard annotations so spaCy
        # can compare its predictions to the truth and compute the loss
        example = Example.from_dict(doc, annots)
        # update() runs the forward pass, computes loss, and adjusts weights
        nlp.update([example], sgd=optimizer, losses=losses, drop=DROP_RATE)

    # (c) Evaluate on dev set
    #     We create Example objects where .predicted is the model's output
    #     and .reference is the gold standard, then use the built-in scorer.
    dev_examples = []
    for text, annots in DEV:
        gold_doc = nlp.make_doc(text)
        example  = Example.from_dict(gold_doc, annots)
        # Run the full pipeline on the raw text to get predictions
        example.predicted = nlp(text)
        dev_examples.append(example)

    scores = Scorer.score_spans(dev_examples, "ents")
    p  = scores["ents_p"]    # precision: of everything the model labelled,
                              #            what fraction was correct?
    r  = scores["ents_r"]    # recall:    of all true entities in the text,
                              #            what fraction did the model find?
    f1 = scores["ents_f"]    # F1:        harmonic mean of P and R — the
                              #            single number that balances both

    print(f"{epoch:>5}  {losses['ner']:>10.2f}  {p:>9.2f}  {r:>6.2f}  {f1:>6.2f}")

    # (d) Save if this is the best epoch so far (by dev F1)
    if f1 >= best_f1:
        best_f1    = f1
        best_epoch = epoch
        nlp.to_disk(OUT_MODEL_DIR)

# ──────────────────────────────────────────────────────────────────────
# 8. DONE
# ──────────────────────────────────────────────────────────────────────
print("─" * 48)
print(f"\nBest dev F1: {best_f1:.2f} (epoch {best_epoch})")
print(f"Model saved to: {OUT_MODEL_DIR.resolve()}")
print(f"\nTo use the model later:")
print(f'    nlp = spacy.load("{OUT_MODEL_DIR}")')
print(f'    doc = nlp("your text here")')
print(f'    for ent in doc.ents:')
print(f'        print(ent.text, ent.label_)')