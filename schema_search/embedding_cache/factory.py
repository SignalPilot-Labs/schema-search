from pathlib import Path
from typing import Dict

from schema_search.embedding_cache.base import BaseEmbeddingCache
from schema_search.embedding_cache.inmemory import InMemoryEmbeddingCache


def create_embedding_cache(config: Dict, cache_dir: Path) -> BaseEmbeddingCache:
    location = config["embedding"]["location"]

    if location == "memory":
        return InMemoryEmbeddingCache(
            cache_dir=cache_dir,
            model_name=config["embedding"]["model"],
            metric=config["embedding"]["metric"],
            batch_size=config["embedding"]["batch_size"],
            show_progress=config["embedding"]["show_progress"],
        )
    else:
        raise ValueError(f"Unsupported embedding location: {location}")
