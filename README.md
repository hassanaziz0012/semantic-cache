# semantic-cache

A small semantic cache for LLM apps in Python. When a new question means the same thing as one you already answered, it returns the stored answer instead of calling the LLM again. Questions are compared by their embeddings (cosine similarity), so "How do I reset my password?" and "I forgot my password. How do I reset it?" share one answer.

It runs free on your own machine: the embedding model is the open-source [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and the LLM call in `app.py` is a stand-in function you swap for your real one.

The full walkthrough, with how the threshold was picked, is in the article: [Semantic caching for LLMs, with a Python example](https://www.hassandev.me/blog/youtube/semantic-caching-stop-paying-for-the-same-answer-3gU_BA_19f0).

## Files

- `semantic_cache.py`: the in-memory cache (embed, lookup, store, with a similarity threshold and an expiry time)
- `app.py`: runs seven support questions through the cache and counts the LLM calls
- `pg_cache.py`: the same cache stored in Postgres with pgvector
- `schema.sql`: the pgvector table and HNSW index
- `expire.sql`: deletes answers older than a day; run it on a schedule

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python app.py
```

The first `pip install` gets the CPU-only PyTorch build, a much smaller download. Skip it if you have an NVIDIA GPU. The first run downloads the model, then prints:

```text
MISS  0.00  How do I reset my password?  ->  LLM answer #1
HIT   0.94  I forgot my password. How do I reset it?  ->  LLM answer #1
HIT   0.89  Help, I can't reset my password.  ->  LLM answer #1
HIT   0.86  how can i change my password  ->  LLM answer #1
MISS  0.44  How do I change my email address?  ->  LLM answer #2
MISS  0.60  Can I update the email on my account?  ->  LLM answer #3
MISS  0.73  My password reset email never arrived  ->  LLM answer #4
7 questions, 4 LLM calls
```

## With Postgres and pgvector

```bash
docker run -d --name semcache-pg -e POSTGRES_PASSWORD=test -p 5432:5432 pgvector/pgvector:0.8.1-pg18
pip install -r requirements-pg.txt
docker exec -i semcache-pg psql -U postgres < schema.sql
```

Then change two lines in `app.py`:

```python
from pg_cache import PgSemanticCache
```

```python
cache = PgSemanticCache("postgresql://postgres:test@localhost:5432/postgres")
```

The first run prints the same as above. The second run is all hits with 0 LLM calls, because the cache is in the database.

Tested on Linux with Python 3.14, PyTorch 2.15 (CPU), sentence-transformers 6.2 and pgvector 0.8.1 on Postgres 18.

## License

MIT
