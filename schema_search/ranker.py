from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from schema_search.chunker import Chunk


class Ranker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bm25 = None
        self.chunks = None

    def build_bm25(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        tokens = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokens)

    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25() first")

        embedding_scores = (embeddings @ query_embedding.T).flatten()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)

        embedding_weight = self.config["search"]["embedding_weight"]
        bm25_weight = self.config["search"]["bm25_weight"]

        combined_scores = (
            embedding_weight * embedding_scores + bm25_weight * bm25_scores_norm
        )

        ranked_indices = combined_scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            results.append(
                (
                    int(idx),
                    float(combined_scores[idx]),
                    float(embedding_scores[idx]),
                    float(bm25_scores_norm[idx]),
                )
            )

        return results

    def get_top_tables_from_chunks(
        self, ranked_chunks: List[Tuple[int, float, float, float]], top_k: int
    ) -> Dict[str, List[int]]:
        if self.chunks is None:
            raise RuntimeError("Chunks not initialized. Call build_bm25() first")

        table_to_chunk_indices = defaultdict(list)

        for chunk_idx, score, emb_score, bm25_score in ranked_chunks:
            chunk = self.chunks[chunk_idx]
            table_to_chunk_indices[chunk.table_name].append(chunk_idx)

        table_scores = {}
        for table_name, chunk_indices in table_to_chunk_indices.items():
            max_score = max(ranked_chunks[idx][1] for idx in chunk_indices)
            table_scores[table_name] = max_score

        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        result = {}
        for table_name, score in top_tables:
            result[table_name] = table_to_chunk_indices[table_name]

        return result
