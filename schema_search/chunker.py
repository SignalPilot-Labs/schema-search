from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import os
import json
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    table_name: str
    content: str
    chunk_id: int
    token_count: int


class Chunker:
    def __init__(self, config: Dict[str, Any]):
        self.max_tokens = config["chunking"]["max_tokens"]
        self.overlap_tokens = config["chunking"]["overlap_tokens"]
        self.use_llm_summary = config["chunking"].get("use_llm_summary", False)
        self.summary_model = config["chunking"].get(
            "summary_model", "claude-sonnet-4-20250514"
        )
        self.llm_client: Optional[Any] = None

        if self.use_llm_summary:
            self._init_llm_client()

    def chunk_metadata(self, metadata_dict: Dict[str, Dict[str, Any]]) -> List[Chunk]:
        chunks = []
        chunk_id = 0

        for table_name, table_info in tqdm(
            metadata_dict.items(), desc="Chunking tables", unit="table"
        ):
            table_chunks = self._chunk_table(table_name, table_info, chunk_id)
            chunks.extend(table_chunks)
            chunk_id += len(table_chunks)

        return chunks

    def _init_llm_client(self):
        try:
            from anthropic import Anthropic
            from dotenv import load_dotenv

            load_dotenv()

            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL")

            if not api_key:
                raise ValueError("LLM_API_KEY not found in environment variables")

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url

            self.llm_client = Anthropic(**kwargs)  # type: ignore
            logger.info(f"Initialized LLM client with model: {self.summary_model}")
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install with: pip install -e '.[llm]'"
            )

    def _chunk_table(
        self, table_name: str, table_info: Dict[str, Any], start_id: int
    ) -> List[Chunk]:
        if self.use_llm_summary:
            markdown = self._generate_llm_summary(table_name, table_info)
        else:
            markdown = self._to_markdown(table_name, table_info)

        lines = markdown.split("\n")

        header = f"Table: {table_name}"
        header_tokens = self._estimate_tokens(header)

        chunks = []
        current_chunk_lines = [header]
        current_tokens = header_tokens
        chunk_id = start_id

        for line in lines[1:]:  # Skip header since we add it manually
            line_tokens = self._estimate_tokens(line)

            if (
                current_tokens + line_tokens > self.max_tokens
                and len(current_chunk_lines) > 1
            ):
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        table_name=table_name,
                        content=chunk_content,
                        chunk_id=chunk_id,
                        token_count=current_tokens,
                    )
                )
                chunk_id += 1

                # Start new chunk with header
                current_chunk_lines = [header]
                current_tokens = header_tokens

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if len(current_chunk_lines) > 1:  # More than just header
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                Chunk(
                    table_name=table_name,
                    content=chunk_content,
                    chunk_id=chunk_id,
                    token_count=current_tokens,
                )
            )

        return chunks

    def _to_markdown(self, table_name: str, table_info: Dict[str, Any]) -> str:
        lines = [f"Table: {table_name}"]

        if table_info.get("columns"):
            col_names = [col["name"] for col in table_info["columns"]]
            # Group columns, ~10 per line to allow chunking on large tables
            cols_per_line = 10
            for i in range(0, len(col_names), cols_per_line):
                batch = col_names[i : i + cols_per_line]
                col_names_str = ", ".join(batch)
                lines.append(f"Columns:{col_names_str}")

        if table_info.get("foreign_keys"):
            related = [fk["referred_table"] for fk in table_info["foreign_keys"]]
            lines.append(f"Related to: {', '.join(related)}")

        if table_info.get("indices"):
            idx_names = [idx["name"] for idx in table_info["indices"] if idx["name"]]
            if idx_names:
                lines.append(f"Indexes: {', '.join(idx_names)}")

        return "\n".join(lines)

    def _generate_llm_summary(self, table_name: str, table_info: Dict[str, Any]) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")

        # Build schema info for the prompt
        schema_json = {
            "table_name": table_name,
            "columns": [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "nullable": col.get("nullable", True),
                    "primary_key": col.get("primary_key", False),
                }
                for col in table_info.get("columns", [])
            ],
            "foreign_keys": [
                {
                    "column": fk["constrained_columns"],
                    "references": f"{fk['referred_table']}.{fk['referred_columns']}",
                }
                for fk in table_info.get("foreign_keys", [])
            ],
            "indices": [
                {"name": idx["name"], "columns": idx["columns"]}
                for idx in table_info.get("indices", [])
                if idx["name"]
            ],
        }

        prompt = f"""Generate a concise 250 tokens or less semantic summary of this database table schema. Focus on:
1. What entity or concept this table represents
2. Key data it stores (main columns)
3. How it relates to other tables
4. Any important constraints or indices

Keep it brief and semantic, optimized for embedding-based search.

Schema:
{json.dumps(schema_json, indent=2)}

Return ONLY the summary text, no preamble."""

        try:
            response = self.llm_client.messages.create(
                model=self.summary_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            summary = response.content[0].text.strip()
            logger.debug(f"Generated LLM summary for {table_name}: {summary[:100]}...")

            # Format as: Table: name\nSummary: ...
            return f"Table: {table_name}\n{summary}"

        except Exception as e:
            logger.warning(
                f"LLM summary failed for {table_name}: {e}. Falling back to markdown."
            )
            return self._to_markdown(table_name, table_info)

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) + len(text) // 4

    def _get_overlap_lines(self, lines: List[str], max_tokens: int) -> List[str]:
        overlap_lines = []
        tokens = 0

        for line in reversed(lines):
            line_tokens = self._estimate_tokens(line)
            if tokens + line_tokens > max_tokens:
                break
            overlap_lines.insert(0, line)
            tokens += line_tokens

        return overlap_lines
