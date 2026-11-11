CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE semantic_cache (
    id bigserial PRIMARY KEY,
    question text NOT NULL,
    answer text NOT NULL,
    embedding vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ON semantic_cache USING hnsw (embedding vector_cosine_ops);
