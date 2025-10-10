from typing import Dict, Optional

from schema_search.chunkers.base import BaseChunker
from schema_search.chunkers.markdown import MarkdownChunker
from schema_search.chunkers.llm import LLMChunker


def create_chunker(
    config: Dict, llm_api_key: Optional[str], llm_base_url: Optional[str]
) -> BaseChunker:
    chunking_config = config["chunking"]
    strategy = chunking_config["strategy"]

    if strategy == "llm":
        return LLMChunker(
            max_tokens=chunking_config["max_tokens"],
            overlap_tokens=chunking_config["overlap_tokens"],
            model=chunking_config["model"],
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    elif strategy == "raw":
        return MarkdownChunker(
            max_tokens=chunking_config["max_tokens"],
            overlap_tokens=chunking_config["overlap_tokens"],
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
