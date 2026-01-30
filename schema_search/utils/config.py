"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from schema_search.utils.utils import lazy_import_check


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "config.yml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_dependencies(config: Dict[str, Any]) -> None:
    """Validate that required dependencies are installed.

    Args:
        config: Configuration dictionary.

    Raises:
        ImportError: If required dependencies are missing.
    """
    strategy = config["search"]["strategy"]
    reranker_model = config["reranker"]["model"]
    chunking_strategy = config["chunking"]["strategy"]

    needs_semantic = strategy in ("semantic", "hybrid") or reranker_model
    if needs_semantic:
        lazy_import_check(
            "sentence_transformers",
            "semantic",
            f"{strategy} search or reranking"
        )

    if chunking_strategy == "llm":
        lazy_import_check("openai", "llm", "LLM-based chunking")
