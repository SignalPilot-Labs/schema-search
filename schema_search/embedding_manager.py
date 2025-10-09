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

    def load_or_generate(self, chunks: List[Chunk], force: bool) -> np.ndarray:
        cache_file = self.cache_dir / "embeddings.npz"
        metadata_file = self.cache_dir / "chunk_metadata.json"
        config_file = self.cache_dir / "cache_config.json"

        if not force and self._is_cache_valid(cache_file, metadata_file, config_file):
            return self._load_from_cache(cache_file, metadata_file)

        return self._generate_and_cache(chunks, cache_file, metadata_file, config_file)

    def _load_from_cache(self, cache_file: Path, metadata_file: Path) -> np.ndarray:
        logger.info(
            f"Loading embeddings from cache ({len(np.load(cache_file)['embeddings'])} embeddings)"
        )
        self.embeddings = np.load(cache_file)["embeddings"]

        with open(metadata_file) as f:
            self.chunk_metadata = json.load(f)

        return self.embeddings

    def _is_cache_valid(
        self, cache_file: Path, metadata_file: Path, config_file: Path
    ) -> bool:
        if not (
            cache_file.exists() and metadata_file.exists() and config_file.exists()
        ):
            return False

        # Check if chunking config has changed
        try:
            with open(config_file) as f:
                cached_config = json.load(f)

            current_config = {
                "use_llm_summary": self.config["chunking"].get(
                    "use_llm_summary", False
                ),
                "max_tokens": self.config["chunking"]["max_tokens"],
                "embedding_model": self.config["embedding"]["model"],
            }

            if cached_config != current_config:
                logger.info("Cache invalidated: chunking config changed")
                return False

            return True
        except Exception as e:
            logger.warning(f"Failed to validate cache config: {e}")
            return False

    def _generate_and_cache(
        self,
        chunks: List[Chunk],
        cache_file: Path,
        metadata_file: Path,
        config_file: Path,
    ) -> np.ndarray:
        self._load_model()
        if self.model is None:
            raise RuntimeError("Embedding model failed to load")

        logger.debug(f"Generating embeddings for {len(chunks)} chunks")
        texts = [chunk.content for chunk in chunks]

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.config["embedding"]["batch_size"],
            normalize_embeddings=self.config["embedding"]["normalize"],
            show_progress_bar=self.config["embedding"]["show_progress"],
        )

        self.chunk_metadata = [
            {
                "table_name": chunk.table_name,
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "content": chunk.content,
            }
            for chunk in chunks
        ]

        np.savez_compressed(cache_file, embeddings=self.embeddings)

        with open(metadata_file, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)

        # Save config for cache validation
        cache_config = {
            "use_llm_summary": self.config["chunking"].get("use_llm_summary", False),
            "max_tokens": self.config["chunking"]["max_tokens"],
            "embedding_model": self.config["embedding"]["model"],
        }
        with open(config_file, "w") as f:
            json.dump(cache_config, f, indent=2)

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
            [query],
            batch_size=self.config["embedding"]["batch_size"],
            normalize_embeddings=self.config["embedding"]["normalize"],
        )

        return query_emb
