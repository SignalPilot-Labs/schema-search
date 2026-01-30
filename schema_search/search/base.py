from typing import List, Optional
from abc import ABC, abstractmethod

from schema_search.types import Chunk, DBSchema, SearchResultItem
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
        db_schema: DBSchema,
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
        limit: int,
        catalogs: Optional[List[str]] = None,
        schemas: Optional[List[str]] = None,
    ) -> List[SearchResultItem]:
        initial_results = self._initial_ranking(
            query, db_schema, chunks, graph_builder, hops
        )

        if catalogs or schemas:
            initial_results = self._filter_results(initial_results, catalogs, schemas)

        if self.reranker is None:
            return initial_results[:limit]

        initial_chunks = []
        for result in initial_results:
            for chunk in chunks:
                if chunk.table_key == result["table"]:
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
                db_schema=db_schema,
                graph_builder=graph_builder,
                hops=hops,
            )
            reranked_results.append(result)

        return reranked_results[:limit]

    def _filter_results(
        self,
        results: List[SearchResultItem],
        catalogs: Optional[List[str]],
        schemas: Optional[List[str]],
    ) -> List[SearchResultItem]:
        """Filter results by catalog and/or schema."""
        filtered = []
        for result in results:
            table_key = result["table"]
            catalog, schema_name = Chunk.parse_schema_key(table_key.rsplit(".", 1)[0])

            if catalogs and catalog not in catalogs:
                continue
            if schemas and schema_name not in schemas:
                continue

            filtered.append(result)
        return filtered

    @abstractmethod
    def _initial_ranking(
        self,
        query: str,
        db_schema: DBSchema,
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        raise NotImplementedError

    def _build_result_item(
        self,
        chunk: Chunk,
        score: float,
        db_schema: DBSchema,
        graph_builder: GraphBuilder,
        hops: int,
    ) -> SearchResultItem:
        table_schema = db_schema[chunk.schema_key][chunk.table_name]

        return {
            "table": chunk.table_key,
            "score": score,
            "schema": table_schema,
            "matched_chunks": [chunk.content],
            "related_tables": list(graph_builder.get_neighbors(chunk.table_key, hops)),
        }
