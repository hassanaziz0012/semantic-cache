import time

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(self, threshold=0.85, ttl_seconds=24 * 60 * 60):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.vectors = []  # one embedding per cached question
        self.entries = []  # (question, answer, stored_at) in the same order

    def embed(self, text):
        # Normalized vectors make the dot product equal to cosine similarity.
        return self.model.encode(text, normalize_embeddings=True)

    def lookup(self, vector):
        if not self.vectors:
            return None, 0.0
        scores = np.array(self.vectors) @ vector
        now = time.time()
        for i, (_, _, stored_at) in enumerate(self.entries):
            if now - stored_at > self.ttl_seconds:
                scores[i] = -1.0  # expired entries never match
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score >= self.threshold:
            return self.entries[best][1], score
        return None, score

    def store(self, question, vector, answer):
        self.vectors.append(vector)
        self.entries.append((question, answer, time.time()))
