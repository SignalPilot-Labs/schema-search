from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from tqdm import tqdm

from schema_search.types import TableSchema, DBSchema


@dataclass
class Chunk:
    schema_name: str
    table_name: str
    content: str
    chunk_id: str
    token_count: int

    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"


class BaseChunker(ABC):
    def __init__(self, max_tokens: int, overlap_tokens: int, show_progress: bool = False):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.show_progress = show_progress

    def chunk_schemas(self, schemas: DBSchema) -> List[Chunk]:
        chunks: List[Chunk] = []

        tables = [
            (schema_name, table_name, table_schema)
            for schema_name, tables in schemas.items()
            for table_name, table_schema in tables.items()
        ]

        iterator = tables
        if self.show_progress:
            iterator = tqdm(tables, desc="Chunking tables", unit="table")

        for schema_name, table_name, table_schema in iterator:
            table_chunks = self._chunk_table(schema_name, table_name, table_schema)
            chunks.extend(table_chunks)

        return chunks

    @abstractmethod
    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        pass

    def _chunk_table(
        self, schema_name: str, table_name: str, table_schema: TableSchema
    ) -> List[Chunk]:
        content = self._generate_content(table_name, table_schema)
        lines = content.split("\n")

        header = f"Schema: {schema_name}\nTable: {table_name}"
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
                        schema_name=schema_name,
                        table_name=table_name,
                        content=chunk_content,
                        chunk_id=f"{schema_name}.{table_name}.{chunk_idx}",
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
                    schema_name=schema_name,
                    table_name=table_name,
                    content=chunk_content,
                    chunk_id=f"{schema_name}.{table_name}.{chunk_idx}",
                    token_count=current_tokens,
                )
            )

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) + len(text) // 4
