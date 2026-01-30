from typing import TypedDict, List, Literal, Optional, Dict, Tuple
from dataclasses import dataclass, field


SearchType = Literal["semantic", "fuzzy", "bm25", "hybrid"]
OutputFormat = Literal["json", "markdown"]


class ColumnInfo(TypedDict):
    name: str
    type: str
    nullable: bool
    default: Optional[str]


class ForeignKeyInfo(TypedDict):
    constrained_columns: List[str]
    referred_schema: str
    referred_table: str
    referred_columns: List[str]


class IndexInfo(TypedDict):
    name: Optional[str]
    columns: List[str]
    unique: bool


class ConstraintInfo(TypedDict):
    name: Optional[str]
    columns: List[str]


class CheckConstraintInfo(TypedDict):
    name: Optional[str]
    sqltext: str


class TableSchema(TypedDict):
    name: str
    schema: str
    primary_keys: List[str]
    columns: Optional[List[ColumnInfo]]
    foreign_keys: Optional[List[ForeignKeyInfo]]
    indices: Optional[List[IndexInfo]]
    unique_constraints: Optional[List[ConstraintInfo]]
    check_constraints: Optional[List[CheckConstraintInfo]]


# Database schema: {schema_key: {table_name: TableSchema}}
# schema_key is "catalog.schema" for Databricks, "schema" for others
DBSchema = Dict[str, Dict[str, TableSchema]]


@dataclass
class Chunk:
    """A chunk of table schema content for indexing and search."""

    catalog: Optional[str]
    schema_name: str
    table_name: str
    content: str
    chunk_id: str
    token_count: int

    @property
    def schema_key(self) -> str:
        """Key for this chunk's schema (catalog.schema or schema)."""
        if self.catalog:
            return f"{self.catalog}.{self.schema_name}"
        return self.schema_name

    @property
    def table_key(self) -> str:
        """Key for this chunk's table (catalog.schema.table or schema.table)."""
        if self.catalog:
            return f"{self.catalog}.{self.schema_name}.{self.table_name}"
        return f"{self.schema_name}.{self.table_name}"

    @staticmethod
    def parse_schema_key(schema_key: str) -> Tuple[Optional[str], str]:
        """Parse schema key into (catalog, schema).

        For Databricks: "samples.bakehouse" -> ("samples", "bakehouse")
        For PostgreSQL: "public" -> (None, "public")
        """
        if "." in schema_key:
            catalog, schema = schema_key.split(".", 1)
            return catalog, schema
        return None, schema_key


class IndexResult(TypedDict):
    tables: int
    chunks: int
    latency_sec: float


class SearchResultItem(TypedDict):
    table: str
    score: float
    schema: TableSchema
    matched_chunks: List[str]
    related_tables: List[str]


@dataclass
class SearchResult:
    """Search result object with rendering capabilities."""

    results: List[SearchResultItem]
    latency_sec: float
    output_format: str = field(default="markdown")

    def __str__(self) -> str:
        """Render results using configured format."""
        from schema_search.renderers.factory import create_renderer
        renderer = create_renderer(self.output_format)
        return renderer.render(self)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "results": self.results,
            "latency_sec": self.latency_sec
        }
