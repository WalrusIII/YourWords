"""
retriever.py — FAISS-backed grammar-rule retriever.

Loads a prebuilt FAISS index of grammar/style rules and returns the rules most
relevant to a given piece of text. This is the "retrieval" in the RAG pipeline:
the grammar-check stage uses these rules to ground its suggestions instead of
relying on the model's memory alone.

Build the index first:  python build_vector_store.py
"""
from __future__ import annotations

import json
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class GrammarRuleRetriever:
    def __init__(self, index_path, meta_path, model_name: str = DEFAULT_MODEL):
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Return up to k rule strings most relevant to `query`."""
        if not query.strip() or not self.rules:
            return []
        emb = self.model.encode([query], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        k = min(k, len(self.rules))
        _scores, idx = self.index.search(emb, k)
        out: List[str] = []
        for i in idx[0]:
            if 0 <= i < len(self.rules):
                rule = self.rules[i]
                if isinstance(rule, dict):
                    category = rule.get("category", "")
                    text = rule.get("rule", "")
                    out.append(f"[{category}] {text}" if category else text)
                else:
                    out.append(str(rule))
        return out
