from typing import List, Optional
from abc import ABC, abstractmethod

from schema_search.types import DBSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder, make_table_key
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
        schemas: DBSchema,
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
                if chunk.qualified_name() == result["table"]:
                    initial_chunks.append(chunk)
                    break

        self.reranker.build(initial_chunks)
        ranked = self.reranker.rank(query)

        reranked_results: List[SearchResultItem] = []
        for chunk_idx, score in ranked[: self.rerank_top_k]:
            chunk = initial_chunks[chunk_idx]
            result = self._build_result_item(
                chunk=chunk,
                score=score,
                schemas=schemas,
                graph_builder=graph_builder,
                hops=hops,
            )
            reranked_results.append(result)

        return reranked_results[:limit]

    @abstractmethod
    def _initial_ranking(
        self,
        query: str,
        schemas: DBSchema,
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        raise NotImplementedError

    def _build_result_item(
        self,
        chunk: Chunk,
        score: float,
        schemas: DBSchema,
        graph_builder: GraphBuilder,
        hops: int,
    ) -> SearchResultItem:
        table_key = make_table_key(chunk.schema_name, chunk.table_name)
        table_schema = schemas[chunk.schema_name][chunk.table_name]

        return {
            "table": table_key,
            "score": score,
            "schema": table_schema,
            "matched_chunks": [chunk.content],
            "related_tables": list(graph_builder.get_neighbors(table_key, hops)),
        }
