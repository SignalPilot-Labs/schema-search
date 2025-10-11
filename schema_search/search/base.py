from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.rankers.base import BaseRanker


class BaseSearchStrategy(ABC):
    def __init__(
        self, reranker: Optional[BaseRanker], initial_top_k: int, rerank_top_k: int
    ):
        self.reranker = reranker
        self.initial_top_k = initial_top_k
        self.rerank_top_k = rerank_top_k

    def search(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
        limit: int,
    ) -> List[SearchResultItem]:
        initial_results = self._initial_ranking(
            query, schemas, chunks, graph_builder, hops
        )

        if self.reranker is None:
            return initial_results[:limit]

        initial_chunks = []
        for result in initial_results:
            for chunk in chunks:
                if chunk.table_name == result["table"]:
                    initial_chunks.append(chunk)
                    break

        self.reranker.build(initial_chunks)
        ranked = self.reranker.rank(query)

        reranked_results: List[SearchResultItem] = []
        for chunk_idx, score in ranked[: self.rerank_top_k]:
            chunk = initial_chunks[chunk_idx]
            result = self._build_result_item(
                table_name=chunk.table_name,
                score=score,
                schema=schemas[chunk.table_name],
                matched_chunks=[chunk.content],
                graph_builder=graph_builder,
                hops=hops,
            )
            reranked_results.append(result)

        return reranked_results[:limit]

    @abstractmethod
    def _initial_ranking(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        pass

    def _build_result_item(
        self,
        table_name: str,
        score: float,
        schema: TableSchema,
        matched_chunks: List[str],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> SearchResultItem:
        return {
            "table": table_name,
            "score": score,
            "schema": schema,
            "matched_chunks": matched_chunks,
            "related_tables": list(graph_builder.get_neighbors(table_name, hops)),
        }
