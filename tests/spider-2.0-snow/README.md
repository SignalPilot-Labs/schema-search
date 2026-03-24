# Spider 2.0-Snow Evaluation

Evaluates **Claude Opus 4.6** on the [Spider 2.0-Snow](https://github.com/xlang-ai/Spider2) Text2SQL benchmark, comparing two modes:

- **vanilla** — Claude has access to `get_schema` tool only
- **mcp** — Claude has access to `get_schema` + `schema_search` tools

Same prompt, same model, same tool loop. The only difference is whether `schema_search` is available. This isolates the effect of semantic schema search on SQL generation quality.

All 547 instances across 152 Snowflake databases.

## Prerequisites

1. **Spider 2.0 Snowflake account** — Request access via the [Snowflake Guideline](https://github.com/xlang-ai/Spider2/blob/main/assets/Snowflake_Guideline.md). Then create `snowflake_credential.json` in this directory:
   ```json
   {
     "username": "<your_username>",
     "password": "<your_generated_token>",
     "account": "RSRSBDK-YDB67606",
     "role": "PARTICIPANT",
     "warehouse": "COMPUTE_WH_PARTICIPANT"
   }
   ```

2. **Anthropic API key** in `tests/.env`:
   ```
   LLM_API_KEY=sk-ant-...
   ```

3. **Dependencies** installed:
   ```bash
   pip install -e ".[snowflake,semantic,mcp]"
   pip install anthropic python-dotenv numpy scipy
   ```

4. **Spider2 repo** already cloned inside this directory (done during setup).

## Usage

### Quick test (5 instances)
```bash
cd tests/spider-2.0-snow
python eval.py --limit 5
```

### Test a specific database
```bash
python eval.py --db-filter GA4 --limit 2
```

### Run only one mode
```bash
python eval.py --modes mcp --limit 10
python eval.py --modes vanilla --limit 10
```

### Full evaluation (both modes, all 547 instances)
```bash
python eval.py
```

## Output

```
results/
  vanilla/           # One .sql file per instance
  mcp/               # One .sql file per instance
  statistical_report.json   # Paired comparison stats
```

## Scoring with Spider 2.0 Evaluation Suite

After generating results, score them using the official evaluator:

```bash
cd Spider2/spider2-snow/evaluation_suite

# Score MCP results
python evaluate.py --mode sql --result_dir ../../../results/mcp

# Score vanilla results
python evaluate.py --mode sql --result_dir ../../../results/vanilla
```

## Statistical Report

When both modes are run, the eval prints a paired statistical comparison:

- **SQL generation rate** per mode (mean, std)
- **McNemar's test** (p-value for paired binary outcomes)
- **Effect size** (Cohen's h)
- **95% confidence interval** for difference in proportions
- **Discordant pairs** (vanilla_wins vs mcp_wins)
- **Tool call stats** (mean, std per mode)
- **Latency** per mode

Report is saved as `results/statistical_report.json`.

## Architecture

| File | Purpose |
|------|---------|
| `eval.py` | Orchestrator — loads dataset, iterates databases, runs both modes, reports stats |
| `agent.py` | Unified Claude API tool loop — same code for both modes, parameterized by tool set |
| `constants.py` | Model name, tool definitions (`TOOLS_VANILLA`, `TOOLS_MCP`), limits |
| `prompt.md` | Shared system prompt (identical for both modes) |

## Experimental Design

Both modes use:
- Same system prompt (`prompt.md`)
- Same model (`claude-opus-4-6`)
- Same agentic loop (`run_agent`)
- Same `get_schema` tool

The only controlled variable:
- **vanilla**: `tools = [get_schema]`
- **mcp**: `tools = [get_schema, schema_search]`
