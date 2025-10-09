from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


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

    def chunk_metadata(self, metadata_dict: Dict[str, Dict[str, Any]]) -> List[Chunk]:
        chunks = []
        chunk_id = 0

        for table_name, table_info in metadata_dict.items():
            table_chunks = self._chunk_table(table_name, table_info, chunk_id)
            chunks.extend(table_chunks)
            chunk_id += len(table_chunks)

        return chunks

    def _chunk_table(self, table_name: str, table_info: Dict[str, Any], start_id: int) -> List[Chunk]:
        markdown = self._to_markdown(table_name, table_info)
        lines = markdown.split("\n")

        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        chunk_id = start_id

        for line in lines:
            line_tokens = self._estimate_tokens(line)

            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(Chunk(
                    table_name=table_name,
                    content=chunk_content,
                    chunk_id=chunk_id,
                    token_count=current_tokens
                ))
                chunk_id += 1

                overlap_lines = self._get_overlap_lines(current_chunk_lines, self.overlap_tokens)
                current_chunk_lines = overlap_lines
                current_tokens = sum(self._estimate_tokens(l) for l in overlap_lines)

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(Chunk(
                table_name=table_name,
                content=chunk_content,
                chunk_id=chunk_id,
                token_count=current_tokens
            ))

        return chunks

    def _to_markdown(self, table_name: str, table_info: Dict[str, Any]) -> str:
        lines = [f"# Table: {table_name}", ""]

        if table_info.get("columns"):
            lines.append("## Columns")
            lines.append("")
            for col in table_info["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                pk_marker = " [PK]" if col["name"] in table_info.get("primary_keys", []) else ""
                default = f" DEFAULT {col['default']}" if col.get("default") else ""
                lines.append(f"- **{col['name']}**: `{col['type']}` {nullable}{pk_marker}{default}")
            lines.append("")

        if table_info.get("primary_keys"):
            lines.append("## Primary Keys")
            lines.append("")
            lines.append(f"- {', '.join(table_info['primary_keys'])}")
            lines.append("")

        if table_info.get("foreign_keys"):
            lines.append("## Foreign Keys")
            lines.append("")
            for fk in table_info["foreign_keys"]:
                constrained = ", ".join(fk["constrained_columns"])
                referred = ", ".join(fk["referred_columns"])
                lines.append(f"- `{constrained}` → `{fk['referred_table']}.{referred}`")
            lines.append("")

        if table_info.get("indices"):
            lines.append("## Indices")
            lines.append("")
            for idx in table_info["indices"]:
                unique = "UNIQUE " if idx["unique"] else ""
                cols = ", ".join(idx["columns"])
                lines.append(f"- {unique}`{idx['name']}` on ({cols})")
            lines.append("")

        return "\n".join(lines)

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
