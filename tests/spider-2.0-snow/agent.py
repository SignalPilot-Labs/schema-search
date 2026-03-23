"""Claude agentic loop for Text2SQL with optional schema-search MCP tools."""

import json
import re
import logging
from collections.abc import Sequence
from typing import Optional

import anthropic
from anthropic.types import ToolParam

from sqlalchemy import Engine, text

from constants import (
    MAX_TOKENS,
    MAX_TOOL_TURNS,
    MODEL_NAME,
    RUN_SQL_MAX_ROWS,
    RUN_SQL_TIMEOUT,
    SEARCH_HOPS,
    SEARCH_LIMIT,
    SQL_CODE_BLOCK_PATTERN,
    TEMPERATURE,
)
from models import AgentResult

logger = logging.getLogger(__name__)


def extract_sql(text: str) -> Optional[str]:
    """Extract SQL from a ```sql code block, or return full text as fallback."""
    match = re.search(SQL_CODE_BLOCK_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    sql_prefixes = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")
    if stripped.upper().startswith(sql_prefixes):
        return stripped
    return None


def _build_user_message(
    instruction: str, external_knowledge: Optional[str], db_id: str
) -> str:
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
    result = search_engine.search(
        query, schemas=schemas, limit=limit, hops=SEARCH_HOPS
    )
    return str(result)


def _handle_get_schema(tool_input: dict, search_engine) -> str:
    """Dispatch get_schema tool call to SchemaSearch.get_schema()."""
    schemas = tool_input.get("schemas")
    result = search_engine.get_schema(schemas=schemas)
    return json.dumps(result, default=str)


def _handle_run_sql(tool_input: dict, engine: Engine) -> str:
    """Execute SQL against Snowflake via raw connection.

    Uses raw DBAPI connection to avoid SQLAlchemy's text() parsing,
    which conflicts with Snowflake's colon notation (value:"field"::TYPE).
    """
    sql_query = tool_input["sql"]
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {RUN_SQL_TIMEOUT}")
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(RUN_SQL_MAX_ROWS)
        total = len(rows)
        extra = cursor.fetchone()
        truncated = extra is not None
        cursor.close()
    finally:
        raw_conn.close()

    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(v) for v in row))
    output = "\n".join(lines)
    if truncated:
        output += f"\n... (showing {total} of more rows)"
    return output


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


def _execute_tool_call(block, search_engine, engine: Engine) -> dict:
    """Execute a single tool call and return the tool_result message."""
    if block.name == "run_sql":
        try:
            result_text = _handle_run_sql(block.input, engine)
        except Exception as e:
            logger.warning("Tool run_sql failed: %s", e)
            result_text = f"Error: {e}"
    else:
        handler = TOOL_HANDLERS.get(block.name)
        if handler is None:
            result_text = f"Error: Unknown tool '{block.name}'"
        else:
            try:
                result_text = handler(block.input, search_engine)
            except Exception as e:
                logger.warning("Tool %s failed: %s", block.name, e)
                result_text = f"Error executing {block.name}: {e}"

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": result_text,
    }


def _process_tool_calls(response, search_engine, engine: Engine) -> list[dict]:
    """Process tool_use blocks and return tool_result messages."""
    return [
        _execute_tool_call(block, search_engine, engine)
        for block in response.content
        if block.type == "tool_use"
    ]


def _build_result(
    instance_id: str,
    sql: str,
    error: Optional[str],
    tool_calls_count: int,
    messages: list,
) -> AgentResult:
    """Build an AgentResult."""
    return AgentResult(
        instance_id=instance_id,
        sql=sql,
        error=error,
        tool_calls_count=tool_calls_count,
        messages=messages,
    )


def run_agent(
    instance_id: str,
    instruction: str,
    external_knowledge: Optional[str],
    search_engine,
    system_prompt: str,
    client: anthropic.Anthropic,
    db_id: str,
    tools: Sequence[ToolParam],
    engine: Engine,
) -> AgentResult:
    """Run Claude with the given tool set (unified loop for both modes).

    Args:
        tools: Tool definitions list — vanilla (get_schema + run_sql)
               or MCP (get_schema + schema_search + run_sql).
        engine: SQLAlchemy engine for run_sql execution.
    """
    user_message = _build_user_message(instruction, external_knowledge, db_id)
    messages: list = [{"role": "user", "content": user_message}]
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
            return _build_result(
                instance_id, sql or "", error, tool_calls_count, messages
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = _process_tool_calls(response, search_engine, engine)
            messages.append({"role": "user", "content": tool_results})
            tool_calls_count += len(tool_results)
            continue

        return _build_result(
            instance_id,
            "",
            f"unexpected_stop_reason:{response.stop_reason}",
            tool_calls_count,
            messages,
        )

    return _build_result(
        instance_id, "", "max_turns_exceeded", tool_calls_count, messages
    )
