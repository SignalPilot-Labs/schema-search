import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from schema_search.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config["embedding"]["cache_dir"])
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
        self.embeddings = None
        self.chunk_metadata = None

    def load_or_generate(self, chunks: List[Chunk]) -> np.ndarray:
        cache_file = self.cache_dir / "embeddings.npz"
        metadata_file = self.cache_dir / "chunk_metadata.json"

        if cache_file.exists() and metadata_file.exists():
            return self._load_from_cache(cache_file, metadata_file)

        return self._generate_and_cache(chunks, cache_file, metadata_file)

    def _load_from_cache(self, cache_file: Path, metadata_file: Path) -> np.ndarray:
        logger.debug(f"Loading embeddings from cache: {cache_file}")
        self.embeddings = np.load(cache_file)["embeddings"]

        with open(metadata_file) as f:
            self.chunk_metadata = json.load(f)

        return self.embeddings

    def _generate_and_cache(
        self, chunks: List[Chunk], cache_file: Path, metadata_file: Path
    ) -> np.ndarray:
        self._load_model()
        if self.model is None:
            raise RuntimeError("Embedding model failed to load")

        logger.debug(f"Generating embeddings for {len(chunks)} chunks")
        texts = [chunk.content for chunk in chunks]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config["embedding"]["normalize"],
            show_progress_bar=True,
        )

        self.chunk_metadata = [
            {
                "table_name": chunk.table_name,
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        np.savez_compressed(cache_file, embeddings=self.embeddings)

        with open(metadata_file, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)

        return self.embeddings

    def _load_model(self):
        if self.model is None:
            model_name = self.config["embedding"]["model"]
            self.model = SentenceTransformer(model_name)

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        if self.model is None:
            raise RuntimeError("Embedding model failed to load")

        query_emb = self.model.encode(
            [query], normalize_embeddings=self.config["embedding"]["normalize"]
        )

        return query_emb
