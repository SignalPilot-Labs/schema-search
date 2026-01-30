import logging
from pathlib import Path
from typing import List, Optional

from sqlalchemy.engine import Engine

from schema_search.chunkers.factory import create_chunker
from schema_search.embedding_cache.factory import create_embedding_cache
from schema_search.embedding_cache.bm25 import BM25Cache
from schema_search.extractors.factory import create_extractor
from schema_search.graph_builder import GraphBuilder
from schema_search.rankers.factory import create_ranker
from schema_search.search.factory import create_search_strategy
from schema_search.types import Chunk, DBSchema, IndexResult, SearchResult, SearchType
from schema_search.utils.cache import load_chunks, load_schema, save_chunks, save_schema, schema_changed
from schema_search.utils.config import load_config, validate_dependencies
from schema_search.utils.utils import setup_logging, time_it


logger = logging.getLogger(__name__)


class SchemaSearch:
    def __init__(
        self,
        engine: Engine,
        config_path: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        setup_logging(self.config)
        validate_dependencies(self.config)

        base_cache_dir = Path(self.config["embedding"]["cache_dir"])
        db_name = engine.url.database or "default"
        cache_dir = base_cache_dir / db_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.schemas: DBSchema = {}
        self.chunks: List[Chunk] = []
        self.cache_dir = cache_dir

        self.extractor = create_extractor(engine, self.config)
        self.chunker = create_chunker(self.config, llm_api_key, llm_base_url)
        self._embedding_cache = None
        self._bm25_cache = None
        self.graph_builder = GraphBuilder(cache_dir)
        self._reranker = None
        self._reranker_config = self.config["reranker"]["model"]
        self._search_strategies = {}

    @time_it
    def index(self, force: bool = False) -> IndexResult:
        logger.info("Starting schema indexing" + (" (force)" if force else ""))

        current_schema = self.extractor.extract()

        has_changed = False
        if not force:
            cached = load_schema(self.cache_dir)
            has_changed = schema_changed(cached, current_schema)
            if has_changed:
                logger.info("Schema change detected; forcing reindex")

        save_schema(self.cache_dir, current_schema)

        effective_force = force or has_changed

        self.schemas = current_schema
        self.graph_builder.build(self.schemas, effective_force)
        self.chunks = self._load_or_generate_chunks(effective_force)
        self._index_force = effective_force

        table_count = sum(len(tables) for tables in self.schemas.values())
        logger.info(
            f"Indexing complete: {table_count} tables, {len(self.chunks)} chunks"
        )
        return {
            "tables": table_count,
            "chunks": len(self.chunks),
            "latency_sec": 0.0,
        }

    def _load_or_generate_chunks(self, force: bool) -> List[Chunk]:
        if not force:
            cached = load_chunks(self.cache_dir)
            if cached is not None:
                return cached

        logger.info("Generating chunks from schemas")
        chunks = self.chunker.chunk_schemas(self.schemas)
        save_chunks(self.cache_dir, chunks)
        return chunks

    def _get_embedding_cache(self):
        if self._embedding_cache is None:
            self._embedding_cache = create_embedding_cache(self.config, self.cache_dir)
        return self._embedding_cache

    def _get_reranker(self):
        if self._reranker is None and self._reranker_config:
            self._reranker = create_ranker(self.config)
        return self._reranker

    @property
    def embedding_cache(self):
        return self._get_embedding_cache()

    @property
    def reranker(self):
        return self._get_reranker()

    def _get_bm25_cache(self):
        if self._bm25_cache is None:
            self._bm25_cache = BM25Cache()
        return self._bm25_cache

    def _ensure_embeddings_loaded(self):
        cache = self._get_embedding_cache()
        if cache.embeddings is None:
            cache.load_or_generate(
                self.chunks, self._index_force, self.config["chunking"]
            )

    def _ensure_bm25_built(self):
        cache = self._get_bm25_cache()
        if cache.bm25 is None:
            logger.info("Building BM25 index")
            cache.build(self.chunks)

    def _get_search_strategy(self, search_type: str):
        if search_type not in self._search_strategies:
            self._search_strategies[search_type] = create_search_strategy(
                self.config,
                self._get_embedding_cache,
                self._get_bm25_cache,
                self._get_reranker,
                search_type,
            )
        return self._search_strategies[search_type]

    def get_schema(
        self,
        catalogs: Optional[List[str]] = None,
        schemas: Optional[List[str]] = None,
    ) -> DBSchema:
        """Get the full schema structure.

        Args:
            catalogs: Optional list of catalog names to include (Databricks only).
            schemas: Optional list of schema names to include.

        Returns:
            Nested dict: {schema_key: {table_name: TableSchema}}
        """
        if not self.schemas:
            raise ValueError("Must call index() before get_schema()")

        if not catalogs and not schemas:
            return self.schemas

        result: DBSchema = {}
        for schema_key, tables in self.schemas.items():
            catalog, schema_name = Chunk.parse_schema_key(schema_key)

            if catalogs and catalog not in catalogs:
                continue
            if schemas and schema_name not in schemas:
                continue

            result[schema_key] = tables

        return result

    @time_it
    def search(
        self,
        query: str,
        catalogs: Optional[List[str]] = None,
        schemas: Optional[List[str]] = None,
        hops: Optional[int] = None,
        limit: Optional[int] = None,
        search_type: Optional[SearchType] = None,
        output_format: Optional[str] = None,
    ) -> SearchResult:
        """Search for tables matching the query.

        Args:
            query: Search query
            catalogs: Optional list of catalog names to search (Databricks only).
            schemas: Optional list of schema names to search.
            hops: Graph traversal hops
            limit: Max results
            search_type: bm25, semantic, fuzzy, or hybrid
            output_format: Output format (markdown or json)

        Returns:
            SearchResult with matching tables
        """
        if not self.chunks:
            raise ValueError("Must call index() before search()")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if hops is None:
            hops = int(self.config["search"]["hops"])
        if limit is None:
            limit = int(self.config["output"]["limit"])

        resolved_format: str = output_format or self.config["output"]["format"]
        resolved_type: str = search_type or self.config["search"]["strategy"]

        logger.debug(f"Searching: {query} (catalogs={catalogs}, schemas={schemas}, hops={hops})")

        if resolved_type in ["semantic", "hybrid"]:
            self._ensure_embeddings_loaded()

        if resolved_type in ["bm25", "hybrid"]:
            self._ensure_bm25_built()

        strategy = self._get_search_strategy(resolved_type)

        results = strategy.search(
            query=query,
            db_schema=self.schemas,
            chunks=self.chunks,
            graph_builder=self.graph_builder,
            hops=hops,
            limit=limit,
            catalogs=catalogs,
            schemas=schemas,
        )

        logger.debug(f"Found {len(results)} results")

        return SearchResult(
            results=results,
            latency_sec=0.0,
            output_format=resolved_format
        )
