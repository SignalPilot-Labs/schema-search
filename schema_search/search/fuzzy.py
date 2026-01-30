from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import Chunk, DBSchema, TableSchema, SearchResultItem
from schema_search.graph_builder import GraphBuilder, make_table_key
from schema_search.rankers.base import BaseRanker


class FuzzySearchStrategy(BaseSearchStrategy):
    def __init__(
        self, initial_top_k: int, rerank_top_k: int, reranker: Optional[BaseRanker]
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)

    def _initial_ranking(
        self,
        query: str,
        db_schema: DBSchema,
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        # (schema_name, table_name, score)
        scored_tables: List[Tuple[str, str, float]] = []

        for schema_name, tables in db_schema.items():
            for table_name, table_schema in tables.items():
                searchable_text = self._build_searchable_text(table_name, table_schema)
                score = fuzz.ratio(query, searchable_text, score_cutoff=0) / 100.0
                scored_tables.append((schema_name, table_name, score))

        scored_tables.sort(key=lambda x: x[2], reverse=True)

        results: List[SearchResultItem] = []
        for schema_name, table_name, score in scored_tables[: self.initial_top_k]:
            table_key = make_table_key(schema_name, table_name)
            table_schema = db_schema[schema_name][table_name]

            result: SearchResultItem = {
                "table": table_key,
                "score": score,
                "schema": table_schema,
                "matched_chunks": [],
                "related_tables": list(graph_builder.get_neighbors(table_key, hops)),
            }
            results.append(result)

        return results

    def _build_searchable_text(self, table_name: str, schema: TableSchema) -> str:
        parts = [table_name]

        indices = schema["indices"]
        if indices:
            for idx in indices:
                if idx["name"]:
                    parts.append(idx["name"])

        return " ".join(parts)
