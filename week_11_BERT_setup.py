import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("All imports successful.")

sentences = [
    # Group 1: Virginia Company — same content, different phrasing/spelling
    "The Virginia Companie did advance traffique beyond the seas.",
    "The Virginia Company did advance traffic beyond the seas.",
    "The Virginia Company promoted overseas trade.",
    
    # Group 2: Merchant activity
    "The merchants of London traded in silks and spices.",
    "London merchants engaged in the silk and spice trade.",
    
    # Group 3: Unrelated (control)
    "The harvest was plentiful this year.",
    "Parliament assembled to debate the new taxation.",
]

# Labels for reference
labels = [
    "VA Co. (archaic spelling)",
    "VA Co. (modern spelling)",
    "VA Co. (paraphrase)",
    "Merchants (version A)",
    "Merchants (version B)",
    "Harvest (unrelated)",
    "Parliament (unrelated)",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

tfidf_sim = cosine_similarity(tfidf_matrix)

# Display as a formatted table
print("TF-IDF Cosine Similarity Matrix:\n")
print(f"{'':>30s}", end="")
for i in range(len(sentences)):
    print(f"  [{i}]", end="")
print()

for i in range(len(sentences)):
    print(f"{labels[i]:>30s}", end="")
    for j in range(len(sentences)):
        print(f" {tfidf_sim[i][j]:5.2f}", end="")
    print()