DELETE FROM semantic_cache WHERE created_at < now() - interval '1 day';
