"""Claude agentic loop for Text2SQL with optional schema-search MCP tools."""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from constants import (
    MAX_TOKENS,
    MAX_TOOL_TURNS,
    MODEL_NAME,
    SEARCH_HOPS,
    SEARCH_LIMIT,
    SQL_CODE_BLOCK_PATTERN,
    TEMPERATURE,
    TOOLS_MCP,
    TOOLS_VANILLA,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent run."""

    instance_id: str
    sql: str
    error: Optional[str]
    tool_calls_count: int
    messages: list = field(default_factory=list)


def extract_sql(text: str) -> Optional[str]:
    """Extract SQL from a ```sql code block, or return full text as fallback."""
    match = re.search(SQL_CODE_BLOCK_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")):
        return stripped
    return None


def _build_user_message(instruction: str, external_knowledge: Optional[str], db_id: str) -> str:
    """Build the user message with question and context."""
    parts = [f"Database: {db_id}", f"Question: {instruction}"]
    if external_knowledge:
        parts.append(f"\nAdditional context:\n{external_knowledge}")
    return "\n\n".join(parts)


def _handle_schema_search(tool_input: dict, search_engine) -> str:
    """Dispatch schema_search tool call to SchemaSearch.search()."""
    query = tool_input["query"]
    schemas = tool_input.get("schemas")
    limit = tool_input.get("limit", SEARCH_LIMIT)
    result = search_engine.search(query, schemas=schemas, limit=limit, hops=SEARCH_HOPS)
    return str(result)


def _handle_get_schema(tool_input: dict, search_engine) -> str:
    """Dispatch get_schema tool call to SchemaSearch.get_schema()."""
    schemas = tool_input.get("schemas")
    result = search_engine.get_schema(schemas=schemas)
    return json.dumps(result, default=str)


TOOL_HANDLERS = {
    "schema_search": _handle_schema_search,
    "get_schema": _handle_get_schema,
}


def _extract_text_from_response(response) -> str:
    """Extract text content from Claude response."""
    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts)


def _process_tool_calls(response, search_engine) -> list:
    """Process tool_use blocks and return tool_result messages."""
    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        handler = TOOL_HANDLERS.get(block.name)
        if handler is None:
            result_text = f"Unknown tool: {block.name}"
        else:
            result_text = handler(block.input, search_engine)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result_text,
        })
    return tool_results


def run_agent(
    instance_id: str,
    instruction: str,
    external_knowledge: Optional[str],
    search_engine,
    system_prompt: str,
    client: anthropic.Anthropic,
    db_id: str,
    tools: list,
) -> AgentResult:
    """Run Claude with the given tool set (unified loop for both modes).

    Args:
        tools: TOOLS_VANILLA (get_schema only) or TOOLS_MCP (get_schema + schema_search).
    """
    user_message = _build_user_message(instruction, external_knowledge, db_id)
    messages = [{"role": "user", "content": user_message}]
    tool_calls_count = 0

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=tools,
            temperature=TEMPERATURE,
        )

        if response.stop_reason == "end_turn":
            text = _extract_text_from_response(response)
            sql = extract_sql(text)
            error = None if sql else "no_sql_extracted"
            return AgentResult(
                instance_id=instance_id,
                sql=sql or "",
                error=error,
                tool_calls_count=tool_calls_count,
                messages=messages,
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = _process_tool_calls(response, search_engine)
            messages.append({"role": "user", "content": tool_results})
            tool_calls_count += len(tool_results)
            continue

        return AgentResult(
            instance_id=instance_id,
            sql="",
            error=f"unexpected_stop_reason:{response.stop_reason}",
            tool_calls_count=tool_calls_count,
            messages=messages,
        )

    return AgentResult(
        instance_id=instance_id,
        sql="",
        error="max_turns_exceeded",
        tool_calls_count=tool_calls_count,
        messages=messages,
    )
