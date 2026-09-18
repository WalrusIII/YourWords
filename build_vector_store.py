"""
build_vector_store.py — build the FAISS index from grammar_rules.json.

Reads grammar_rules.json (a list of {"category": ..., "rule": ...} objects),
embeds each rule, and writes:
    grammar_rules.index       the FAISS index
    grammar_rules_meta.json   the rules, aligned to index positions

Run once before starting the app:  python build_vector_store.py
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
RULES_FILE = HERE / "grammar_rules.json"
INDEX_FILE = HERE / "grammar_rules.index"
META_FILE = HERE / "grammar_rules_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        rules = json.load(f)

    texts = [r["rule"] if isinstance(r, dict) else str(r) for r in rules]
    print(f"Embedding {len(texts)} grammar rules with {MODEL_NAME} ...")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    print(f"Wrote {INDEX_FILE.name} and {META_FILE.name} ({index.ntotal} vectors).")


if __name__ == "__main__":
    main()
