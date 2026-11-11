import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


class PgSemanticCache:
    def __init__(self, dsn, threshold=0.85):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = threshold
        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)

    def embed(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    def lookup(self, vector):
        row = self.conn.execute(
            """
            SELECT answer, 1 - (embedding <=> %(v)s) AS similarity
            FROM semantic_cache
            ORDER BY embedding <=> %(v)s
            LIMIT 1
            """,
            {"v": vector},
        ).fetchone()
        if row is None:
            return None, 0.0
        answer, score = row
        if score >= self.threshold:
            return answer, score
        return None, score

    def store(self, question, vector, answer):
        self.conn.execute(
            "INSERT INTO semantic_cache (question, answer, embedding) VALUES (%s, %s, %s)",
            (question, answer, vector),
        )
