from typing import Dict, List, Optional

import numpy as np
from rapidfuzz import fuzz

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.embedding_cache import BaseEmbeddingCache
from schema_search.rankers.base import BaseRanker


class HybridSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        embedding_cache: BaseEmbeddingCache,
        initial_top_k: int,
        rerank_top_k: int,
        reranker: Optional[BaseRanker],
        semantic_weight: float,
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)
        assert 0 <= semantic_weight <= 1, "semantic_weight must be between 0 and 1"
        self.embedding_cache = embedding_cache
        self.semantic_weight = semantic_weight
        self.fuzzy_weight = 1 - semantic_weight

    def _initial_ranking(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        query_embedding = self.embedding_cache.encode_query(query)
        semantic_scores = self.embedding_cache.compute_similarities(query_embedding)

        table_to_chunks = {}
        for idx, chunk in enumerate(chunks):
            if chunk.table_name not in table_to_chunks:
                table_to_chunks[chunk.table_name] = []
            table_to_chunks[chunk.table_name].append(idx)

        fuzzy_scores = np.zeros(len(chunks))
        for table_name, schema in schemas.items():
            searchable_text = self._build_searchable_text(table_name, schema)
            score = fuzz.ratio(query, searchable_text, score_cutoff=0) / 100.0
            if table_name in table_to_chunks:
                for idx in table_to_chunks[table_name]:
                    fuzzy_scores[idx] = score

        semantic_min = semantic_scores.min()
        semantic_max = semantic_scores.max()
        semantic_range = semantic_max - semantic_min
        if semantic_range > 0:
            semantic_scores_norm = (semantic_scores - semantic_min) / semantic_range
        else:
            semantic_scores_norm = np.zeros_like(semantic_scores)

        fuzzy_min = fuzzy_scores.min()
        fuzzy_max = fuzzy_scores.max()
        fuzzy_range = fuzzy_max - fuzzy_min
        if fuzzy_range > 0:
            fuzzy_scores_norm = (fuzzy_scores - fuzzy_min) / fuzzy_range
        else:
            fuzzy_scores_norm = np.zeros_like(fuzzy_scores)

        hybrid_scores = (
            self.semantic_weight * semantic_scores_norm
            + self.fuzzy_weight * fuzzy_scores_norm
        )

        top_indices = hybrid_scores.argsort()[::-1][: self.initial_top_k]

        results: List[SearchResultItem] = []
        for idx in top_indices:
            chunk = chunks[idx]
            result = self._build_result_item(
                table_name=chunk.table_name,
                score=float(hybrid_scores[idx]),
                schema=schemas[chunk.table_name],
                matched_chunks=[chunk.content],
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        return results

    def _build_searchable_text(self, table_name: str, schema: TableSchema) -> str:
        parts = [table_name]
        if schema["indices"]:
            for idx in schema["indices"]:
                parts.append(idx["name"])
        return " ".join(parts)
