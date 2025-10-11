from typing import List, Tuple
import logging

from sentence_transformers import CrossEncoder

from schema_search.chunkers import Chunk
from schema_search.rankers.base import BaseRanker

logger = logging.getLogger(__name__)


class CrossEncoderRanker(BaseRanker):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = None

    def _load_model(self) -> CrossEncoder:
        if self.model is None:
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder: {self.model_name}")
        return self.model

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        logger.debug(f"Initialized CrossEncoder reranker with {len(chunks)} chunks")

    def rank(self, query: str) -> List[Tuple[int, float]]:
        model = self._load_model()
        pairs = [(query, chunk.content) for chunk in self.chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked_indices = scores.argsort()[::-1]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]
