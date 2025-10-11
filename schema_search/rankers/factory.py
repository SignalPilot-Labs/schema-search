from typing import Dict, Optional

from schema_search.rankers.base import BaseRanker
from schema_search.rankers.cross_encoder import CrossEncoderRanker


def create_ranker(config: Dict) -> Optional[BaseRanker]:
    reranker_model = config["reranker"]["model"]
    if reranker_model is None:
        return None
    return CrossEncoderRanker(model_name=reranker_model)
