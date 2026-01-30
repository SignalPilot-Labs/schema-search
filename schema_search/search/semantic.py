from typing import List, Optional

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import Chunk, DBSchema, SearchResultItem
from schema_search.graph_builder import GraphBuilder
from schema_search.embedding_cache.base import BaseEmbeddingCache
from schema_search.rankers.base import BaseRanker


class SemanticSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        embedding_cache: BaseEmbeddingCache,
        initial_top_k: int,
        rerank_top_k: int,
        reranker: Optional[BaseRanker],
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)
        self.embedding_cache = embedding_cache

    def _initial_ranking(
        self,
        query: str,
        db_schema: DBSchema,
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        query_embedding = self.embedding_cache.encode_query(query)
        embedding_scores = self.embedding_cache.compute_similarities(query_embedding)
        top_indices = embedding_scores.argsort()[::-1][: self.initial_top_k]

        results: List[SearchResultItem] = []
        for idx in top_indices:
            chunk = chunks[idx]
            result = self._build_result_item(
                chunk=chunk,
                score=float(embedding_scores[idx]),
                db_schema=db_schema,
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        return results
