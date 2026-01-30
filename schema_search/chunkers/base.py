from typing import List, Optional
from abc import ABC, abstractmethod

from tqdm import tqdm

from schema_search.types import Chunk, TableSchema, DBSchema


class BaseChunker(ABC):
    """Base class for schema chunkers."""

    def __init__(self, max_tokens: int, overlap_tokens: int, show_progress: bool = False):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.show_progress = show_progress

    def chunk_schemas(self, schemas: DBSchema) -> List[Chunk]:
        """Chunk all schemas into searchable pieces."""
        chunks: List[Chunk] = []

        tables = [
            (schema_key, table_name, table_schema)
            for schema_key, tables in schemas.items()
            for table_name, table_schema in tables.items()
        ]

        iterator = tables
        if self.show_progress:
            iterator = tqdm(tables, desc="Chunking tables", unit="table")

        for schema_key, table_name, table_schema in iterator:
            catalog, schema_name = Chunk.parse_schema_key(schema_key)
            table_chunks = self._chunk_table(catalog, schema_name, table_name, table_schema)
            chunks.extend(table_chunks)

        return chunks

    @abstractmethod
    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        pass

    def _chunk_table(
        self,
        catalog: Optional[str],
        schema_name: str,
        table_name: str,
        table_schema: TableSchema,
    ) -> List[Chunk]:
        """Chunk a single table's schema into searchable pieces."""
        content = self._generate_content(table_name, table_schema)
        lines = content.split("\n")

        schema_key = f"{catalog}.{schema_name}" if catalog else schema_name
        header = f"Schema: {schema_key}\nTable: {table_name}"
        header_tokens = self._estimate_tokens(header)

        chunks: List[Chunk] = []
        current_chunk_lines = [header]
        current_tokens = header_tokens
        chunk_idx = 0

        for line in lines[1:]:
            line_tokens = self._estimate_tokens(line)

            if (
                current_tokens + line_tokens > self.max_tokens
                and len(current_chunk_lines) > 1
            ):
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        catalog=catalog,
                        schema_name=schema_name,
                        table_name=table_name,
                        content=chunk_content,
                        chunk_id=f"{schema_key}.{table_name}.{chunk_idx}",
                        token_count=current_tokens,
                    )
                )
                chunk_idx += 1

                current_chunk_lines = [header]
                current_tokens = header_tokens

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if len(current_chunk_lines) > 1:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                Chunk(
                    catalog=catalog,
                    schema_name=schema_name,
                    table_name=table_name,
                    content=chunk_content,
                    chunk_id=f"{schema_key}.{table_name}.{chunk_idx}",
                    token_count=current_tokens,
                )
            )

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) + len(text) // 4
