from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.rankers.base import BaseRanker


class BM25SearchStrategy(BaseSearchStrategy):
    def __init__(
        self, initial_top_k: int, rerank_top_k: int, reranker: Optional[BaseRanker]
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)

    def _initial_ranking(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        texts = [chunk.content for chunk in chunks]
        tokens = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokens)

        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        top_indices = bm25_scores.argsort()[::-1][: self.initial_top_k]

        results: List[SearchResultItem] = []
        for idx in top_indices:
            chunk = chunks[idx]
            result = self._build_result_item(
                table_name=chunk.table_name,
                score=float(bm25_scores[idx]),
                schema=schemas[chunk.table_name],
                matched_chunks=[chunk.content],
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        return results
