from typing import Dict

from schema_search.search.semantic import SemanticSearchStrategy
from schema_search.search.fuzzy import FuzzySearchStrategy
from schema_search.search.base import BaseSearchStrategy
from schema_search.embedding_cache import BaseEmbeddingCache
from schema_search.rankers import create_ranker


def create_semantic_strategy(
    config: Dict, embedding_cache: BaseEmbeddingCache
) -> SemanticSearchStrategy:
    reranker = create_ranker(config)
    return SemanticSearchStrategy(
        embedding_cache=embedding_cache,
        initial_top_k=config["search"]["initial_top_k"],
        rerank_top_k=config["search"]["rerank_top_k"],
        reranker=reranker,
    )


def create_fuzzy_strategy() -> FuzzySearchStrategy:
    return FuzzySearchStrategy()
