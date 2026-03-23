"""Score generated SQL against Spider2 gold results via execution."""

import json
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd
import snowflake.connector

from constants import CREDENTIAL_PATH, SPIDER2_SNOW_DIR

EVAL_SUITE_DIR = SPIDER2_SNOW_DIR / "evaluation_suite"
GOLD_DIR = EVAL_SUITE_DIR / "gold"
GOLD_RESULT_DIR = GOLD_DIR / "exec_result"
EVAL_STANDARD_PATH = GOLD_DIR / "spider2snow_eval.jsonl"
METADATA_PATH = SPIDER2_SNOW_DIR / "spider2-snow.jsonl"
SCORE_TIMEOUT = 60


def _load_jsonl_to_dict(path: Path) -> dict:
    """Load JSONL file into dict keyed by instance_id."""
    data = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            data[item["instance_id"]] = item
    return data


def _compare_pandas_table(
    pred: pd.DataFrame, gold: pd.DataFrame,
    condition_cols: list, ignore_order: bool,
) -> int:
    """Compare predicted and gold DataFrames. Returns 1 if match, 0 otherwise."""
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]
        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if abs(float(a) - float(b)) > tol:
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred.transpose().values.tolist()

    for gold_vec in t_gold_list:
        if not any(vectors_match(gold_vec, pred_vec, ignore_order_=ignore_order) for pred_vec in t_pred_list):
            return 0
    return 1


def _compare_multi(
    pred: pd.DataFrame, multi_gold: list,
    multi_condition_cols: list, ignore_order: bool,
) -> int:
    """Try matching pred against multiple gold DataFrames."""
    if not multi_condition_cols or multi_condition_cols == [[]] or multi_condition_cols == [None]:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(s, list) for s in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if _compare_pandas_table(pred, gold, multi_condition_cols[i], ignore_order):
            return 1
    return 0


def _build_connection_kwargs(credential: dict) -> dict:
    """Build snowflake.connector kwargs from credential dict.

    Handles mapping 'username' -> 'user' and adding PAT authenticator.
    """
    kwargs = {}
    kwargs["user"] = credential.get("user", credential.get("username", ""))
    kwargs["account"] = credential["account"]
    if "warehouse" in credential:
        kwargs["warehouse"] = credential["warehouse"]
    if "role" in credential:
        kwargs["role"] = credential["role"]
    if "token" in credential:
        kwargs["token"] = credential["token"]
        kwargs["authenticator"] = credential.get(
            "authenticator", "programmatic_access_token"
        )
    elif "password" in credential:
        kwargs["password"] = credential["password"]
    kwargs["session_parameters"] = {
        "STATEMENT_TIMEOUT_IN_SECONDS": SCORE_TIMEOUT,
    }
    return kwargs


def _execute_and_score(
    instance_id: str, sql_path: Path, db_id: str,
    eval_standard: dict, credential: dict, temp_dir: Path,
) -> dict:
    """Execute SQL and compare result against gold."""
    sql_text = sql_path.read_text().strip()
    if not sql_text:
        return {"instance_id": instance_id, "score": 0, "error": "empty_sql"}

    # Extract SQL from code block if present
    match = re.search(r"```sql\s*(.*?)\s*```", sql_text, re.DOTALL)
    if match:
        sql_text = match.group(1).strip()

    # Execute against Snowflake
    connection_kwargs = _build_connection_kwargs(credential)

    try:
        conn = snowflake.connector.connect(database=db_id, **connection_kwargs)
        cursor = conn.cursor()
        cursor.execute(sql_text)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        pred_df = pd.DataFrame(results, columns=columns)
        cursor.close()
        conn.close()
    except Exception as e:
        return {"instance_id": instance_id, "score": 0, "error": str(e)}

    if pred_df.empty:
        return {"instance_id": instance_id, "score": 0, "error": "empty_result"}

    # Compare against gold
    eval_entry = eval_standard.get(instance_id, {})
    condition_cols = eval_entry.get("condition_cols", [])
    ignore_order = eval_entry.get("ignore_order", False)

    # Find gold CSV files
    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    all_gold_files = os.listdir(GOLD_RESULT_DIR)
    csv_files = sorted(f for f in all_gold_files if pattern.match(f))

    base_path = GOLD_RESULT_DIR / f"{instance_id}.csv"
    try:
        if base_path.exists():
            gold_df = pd.read_csv(base_path)
            score = _compare_pandas_table(pred_df, gold_df, condition_cols, ignore_order)
        elif csv_files:
            gold_dfs = [pd.read_csv(GOLD_RESULT_DIR / f) for f in csv_files]
            score = _compare_multi(pred_df, gold_dfs, condition_cols, ignore_order)
        else:
            return {"instance_id": instance_id, "score": 0, "error": "no_gold_file"}
    except Exception as e:
        return {"instance_id": instance_id, "score": 0, "error": f"compare_error:{e}"}

    error = None if score == 1 else "result_mismatch"
    return {"instance_id": instance_id, "score": score, "error": error}


def score_mode(
    mode: str, output_dir: Path, credential: dict,
) -> Dict[str, dict]:
    """Score all SQL files for a mode. Returns dict of instance_id -> result."""
    result_dir = output_dir / mode
    if not result_dir.exists():
        return {}

    eval_standard = _load_jsonl_to_dict(EVAL_STANDARD_PATH)
    metadata = _load_jsonl_to_dict(METADATA_PATH)

    sql_files = sorted(result_dir.glob("*.sql"))
    if not sql_files:
        return {}

    temp_dir = Path(tempfile.mkdtemp(prefix=f"score_{mode}_"))
    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for sql_path in sql_files:
            iid = sql_path.stem
            if iid not in eval_standard:
                continue
            db_id = metadata.get(iid, {}).get("db_id", "")
            futures[executor.submit(
                _execute_and_score, iid, sql_path, db_id,
                eval_standard, credential, temp_dir,
            )] = iid

        for future in as_completed(futures):
            result = future.result()
            results[result["instance_id"]] = result

    shutil.rmtree(temp_dir, ignore_errors=True)
    return results
