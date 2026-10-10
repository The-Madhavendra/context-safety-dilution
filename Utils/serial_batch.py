"""
serial_batch.py
===============
Serial batch evaluation script to detect and analyze scoring variance across repeated runs
of safety benchmark prompts on Gemini (evaluated via Groq judge).

Workflow:
1. Baseline Runs: Executes an initial set of runs (default: 3) across the full dataset
   (e.g., 100 prompts) using Inspect AI with real-time terminal progress.
2. Variance Detection: Identifies all prompts exhibiting score differences across any run
   (i.e., len(set(scores)) > 1).
3. CSV Logging: Creates a single CSV session log in `Experiment A2/varied_response_logs/`
   recording prompt ID, category, prompt text, and scores for each iteration (run_1, run_2, run_3...).
4. Interactive Iterative Re-Evaluation: Prompts the user (or uses --additional-runs) to execute
   further eval runs specifically on the varied dataset subset, adding columns (run_4, run_5...)
   to the same CSV and updating it dynamically.

Run via CLI:
  python .\\Utils\\serial_batch.py
  python .\\Utils\\serial_batch.py --padding 0k --position middle --initial-runs 3
  python .\\Utils\\serial_batch.py --additional-runs 2 --limit 10
  python .\\Utils\\serial_batch.py --printAPI
"""

import os
import sys
import csv
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Inspect AI imports
from inspect_ai import Task, eval as inspect_run_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, TaskState
from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.model import GenerateConfig

# Shared evaluation utilities and paths
from Utils.eval_funcs import (
    BASE_HARMFUL_PROMPTS_PATH,
    CONTEXTS_CHUNKED_PATH,
    MASTER_PROMPT_PATH,
    EXPERIMENT_A2_DIR,
    CUSTOM_LOGS_DIR,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MASTER_CSV_NAME,
    build_mapped_dataset,
    call_groq_judge,
    call_judge,
    call_gemini_judge,
    save_custom_log,
    append_to_master_csv,
    generate_eval_id,
    find_inspect_log_file,
    update_master_csv_varied_log,
    gemini_dynamic_solver,
    get_inspect_serial_semaphore
)

# Quota management
from Utils.api_management import (
    pre_flight_allocate,
    print_available_api_keys,
    QuotaExceededError
)


# ==============================================================================
# CONFIGURABLE VARIABLES & DEFAULTS
# ==============================================================================
DEFAULT_PADDING: str = "none"                   # Context padding: "none" ("0k"), "1k", "2k", "5k", "10k", "20k", "50k"
DEFAULT_POSITION: str = "middle"                # Harmful prompt position: "start", "middle", "end"
DEFAULT_MODEL: str = DEFAULT_GEMINI_MODEL       # Target model: "google/gemini-3.5-flash-lite"
DEFAULT_JUDGE: str = DEFAULT_JUDGE_MODEL        # Safety judge model: "google/gemini-3.1-flash-lite"
DEFAULT_INITIAL_RUNS: int = 3                   # Initial baseline runs to assess variance
DEFAULT_ADDITIONAL_RUNS: Optional[int] = None   # Additional runs on varied set (None = prompt interactively)
DEFAULT_LIMIT: Optional[int] = None             # Sample limit (None = full dataset, e.g. 100 prompts)
MAX_CONNECTIONS: int = DEFAULT_MAX_CONNECTIONS  # 1 = strictly serial execution (one sample at a time)

EXPERIMENT_NAME: str = "A2"
VARIED_LOGS_DIR: Path = EXPERIMENT_A2_DIR / "varied_response_logs"
INSPECT_LOGS_DIR: Path = EXPERIMENT_A2_DIR / "logs"
RUN_COLUMN_PREFIX: str = "run_"                 # Column name prefix for runs: "run_1", "run_2", ...


# ==============================================================================
# 1. HELPER FUNCTIONS: NORMALIZATION & DATASET PREPARATION
# ==============================================================================
def normalize_padding(padding: str) -> str:
    """Normalizes padding string: 'none', '0', '0k' -> '0k', '1k' -> '1k', etc."""
    p = str(padding).lower().strip()
    return "0k" if p in ("none", "0", "0k") else p


def normalize_position(position: str) -> str:
    """Normalizes prompt position: 'start', 'middle', 'end'."""
    pos = str(position).lower().strip()
    return pos if pos in ("start", "middle", "end") else "middle"


def load_full_dataset(padding: str, position: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads and formats the harmful prompts dataset with the requested context padding and position."""
    norm_pad = normalize_padding(padding)
    norm_pos = normalize_position(position)
    dataset = build_mapped_dataset(
        harmful_prompts_file=BASE_HARMFUL_PROMPTS_PATH,
        contexts_chunked_file=CONTEXTS_CHUNKED_PATH,
        padding=norm_pad,
        position=norm_pos
    )
    if limit is not None and limit > 0:
        return dataset[:limit]
    return dataset


# ==============================================================================
# 2. CUSTOM INSPECT SCORER (GROQ 0-4 RUBRIC)
# ==============================================================================
@scorer(metrics=[mean()])
def serial_batch_scorer(
    judge_model: str = DEFAULT_JUDGE,
    master_prompt_path: Path = MASTER_PROMPT_PATH,
    scores_collector: Optional[Dict[str, int]] = None,
    results_collector: Optional[List[Dict[str, Any]]] = None
):
    """
    Inspect AI custom scorer for serial batch runs.
    Sends raw harmful request and model response to Groq judge for grading (0-4).
    Collects live scores in memory and detailed sample records for custom_logs and master CSV.
    """
    with open(master_prompt_path, "r", encoding="utf-8") as f:
        master_prompt_template = f.read()

    async def score(state: TaskState, target: Target) -> Score:
        sem = get_inspect_serial_semaphore()
        try:
            # Record Google API quota immediately after this sample run completes
            if not state.metadata.get("_gemini_quota_recorded"):
                state.metadata["_gemini_quota_recorded"] = True
                try:
                    from Utils.api_management import record_sample_usage
                    record_sample_usage("google")
                except Exception as e:
                    print(f"[Warning] Failed to update Google quota: {e}")

            harmful_prompt = state.metadata.get("raw_prompt", str(state.input))
            model_response = state.output.completion if state.output else ""

            judge_res = await asyncio.to_thread(
                call_judge,
                harmful_request=harmful_prompt,
                model_response=model_response,
                master_prompt_template=master_prompt_template,
                judge_model=judge_model
            )

            # Strict pacing delay between samples to prevent burst rate limit errors
            await asyncio.sleep(2.0)

            score_val = judge_res.get("score", -1)
            reason = judge_res.get("reason", "")
            evidence = judge_res.get("evidence", "")

            sample_id_str = str(state.metadata.get("id", state.sample_id))
            if scores_collector is not None:
                scores_collector[sample_id_str] = score_val

            if results_collector is not None:
                results_collector.append({
                    "id": sample_id_str,
                    "category": state.metadata.get("category", ""),
                    "context_category": state.metadata.get("context_category", ""),
                    "padding": state.metadata.get("padding", "0k"),
                    "position": state.metadata.get("position", "middle"),
                    "raw_prompt": harmful_prompt,
                    "gemini_response": model_response,
                    "judge_score": score_val,
                    "judge_reason": reason,
                    "judge_evidence": evidence
                })

            return Score(
                value=score_val,
                answer=model_response,
                explanation=f"[Score: {score_val}/4] {reason}",
                metadata={
                    "id": sample_id_str,
                    "rubric_score": score_val,
                    "reason": reason,
                    "evidence": evidence,
                    "raw_prompt": harmful_prompt,
                    "category": state.metadata.get("category", "")
                }
            )
        finally:
            if state.metadata.get("_serial_lock_acquired"):
                sem.release()
                state.metadata["_serial_lock_acquired"] = False

    return score


def extract_score_and_metadata(sample: Sample) -> Tuple[int, Dict[str, Any]]:
    """
    Extracts the integer rubric score and metadata dict from an Inspect Sample.
    Prioritizes sample.scores (dict/list) over deprecated sample.score.
    """
    score_obj = None
    if hasattr(sample, "scores") and sample.scores:
        if isinstance(sample.scores, dict):
            score_obj = next(iter(sample.scores.values()), None)
        elif isinstance(sample.scores, list) and len(sample.scores) > 0:
            score_obj = sample.scores[0]
    if score_obj is None and hasattr(sample, "score"):
        score_obj = sample.score

    if not score_obj:
        return -1, {}

    score_meta = score_obj.metadata if isinstance(score_obj.metadata, dict) else {}
    val = score_meta.get("rubric_score", score_obj.value if score_obj.value is not None else -1)
    try:
        val = int(val)
    except (ValueError, TypeError):
        val = -1
    return val, score_meta


# ==============================================================================
# 3. SINGLE INSPECT RUN EXECUTION
# ==============================================================================
def execute_inspect_run(
    items_to_evaluate: List[Dict[str, Any]],
    run_label: str,
    run_index: int,
    padding: str,
    position: str,
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_JUDGE,
    max_connections: int = MAX_CONNECTIONS,
    log_dir: Path = INSPECT_LOGS_DIR,
    varied_log_rel_path: str = "",
    varied_response_summary: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Executes an evaluation run using Inspect AI for the provided dataset items.
    Handles Best-Fit API key allocation pre-flight check, renders live terminal progress,
    saves detailed JSON to custom_logs/, appends summary row to master_eval_results.csv,
    and returns (scores_by_id, run_record).
    """
    norm_pad = normalize_padding(padding)
    norm_pos = normalize_position(position)
    num_items = len(items_to_evaluate)

    # Pre-flight quota check and key activation (does not pre-deduct)
    print(f"\n>>> [RUN {run_index}] Checking API Quota for {num_items} requests...")
    try:
        pre_flight_allocate(needed_requests=num_items, model_name="gemini-3.5-flash-lite", deduct=False)
    except Exception as e:
        print(f"[Warning] Target model preflight allocation check: {e}")

    try:
        clean_judge_name = judge_model.replace("google/", "").strip()
        if "gemini" in clean_judge_name:
            pre_flight_allocate(needed_requests=num_items, model_name=clean_judge_name, deduct=False)
        elif "gpt" in clean_judge_name or "groq" in clean_judge_name:
            pre_flight_allocate(needed_requests=num_items, model_name="openai/gpt-oss-120b", deduct=False)
    except Exception as e:
        print(f"[Warning] Judge model preflight allocation check: {e}")

    task_name = f"{EXPERIMENT_NAME}_{norm_pad}_{norm_pos}_{run_label}"

    samples = [
        Sample(
            input=item["padded_prompt"],
            id=item["id"],
            metadata={
                "id": item["id"],
                "raw_prompt": item["raw_prompt"],
                "category": item["category"],
                "context_category": item.get("context_category", ""),
                "padding": norm_pad,
                "position": norm_pos,
                "run_label": run_label,
                "judge_model": judge_model
            }
        )
        for item in items_to_evaluate
    ]

    live_scores: Dict[str, int] = {}
    live_sample_results: List[Dict[str, Any]] = []

    task = Task(
        name=task_name,
        display_name=task_name,
        dataset=samples,
        solver=[gemini_dynamic_solver(model)],
        scorer=serial_batch_scorer(
            judge_model=judge_model,
            scores_collector=live_scores,
            results_collector=live_sample_results
        ),
        config=GenerateConfig(max_connections=max_connections)
    )

    print(f">>> [RUN {run_index}] Starting Inspect AI Eval Task: '{task_name}' ({num_items} prompts)")

    log_dir.mkdir(parents=True, exist_ok=True)
    eval_results = inspect_run_eval(
        tasks=[task],
        model=model,
        max_connections=max_connections,
        max_samples=max_connections,
        log_dir=str(log_dir)
    )

    # Extract score mapping: first use live_scores, with fallback to log deserialization
    scores_by_id: Dict[str, int] = dict(live_scores)

    if len(scores_by_id) < num_items and eval_results and len(eval_results) > 0:
        eval_log = eval_results[0]
        samples_list = eval_log.samples
        if samples_list is None and hasattr(eval_log, "location") and eval_log.location:
            try:
                from inspect_ai.log import read_eval_log
                loaded_log = read_eval_log(eval_log.location)
                samples_list = loaded_log.samples
            except Exception:
                pass
        if samples_list:
            for s in samples_list:
                sid = str(s.id)
                if sid not in scores_by_id:
                    val, sm = extract_score_and_metadata(s)
                    scores_by_id[sid] = val
                    if not any(r["id"] == sid for r in live_sample_results):
                        raw_p = str(getattr(s, "input", ""))
                        live_sample_results.append({
                            "id": sid,
                            "category": str(sm.get("category", "")),
                            "context_category": str(sm.get("context_category", "")),
                            "padding": norm_pad,
                            "position": norm_pos,
                            "raw_prompt": raw_p,
                            "gemini_response": str(getattr(getattr(s, "output", None), "completion", "")),
                            "judge_score": val,
                            "judge_reason": str(sm.get("reason", "")),
                            "judge_evidence": str(sm.get("evidence", ""))
                        })

    # Resolve relative path to matching Inspect AI .eval log in logs/
    inspect_rel_path = ""
    if eval_results and len(eval_results) > 0 and hasattr(eval_results[0], "location") and eval_results[0].location:
        loc = str(eval_results[0].location).replace("file://", "").strip()
        inspect_rel_path = f"logs/{Path(loc).name}"
    else:
        inspect_rel_path = find_inspect_log_file(log_dir, task_name=task_name)

    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    ts_slug = now_dt.strftime("%Y%m%d_%H%M%S")
    eval_id = generate_eval_id(norm_pad, norm_pos, run_label, timestamp_slug=ts_slug)

    custom_log_filename = f"{task_name}_{ts_slug}.json"
    custom_log_path = CUSTOM_LOGS_DIR / custom_log_filename
    custom_log_rel = f"custom_logs/{custom_log_filename}"

    meta = {
        "eval_id": eval_id,
        "timestamp": now_str,
        "run_label": run_label,
        "run_index": run_index,
        "task_name": task_name,
        "padding": norm_pad,
        "position": norm_pos,
        "gemini_model": model,
        "judge_model": judge_model,
        "total_evaluated": len(live_sample_results),
        "master_csv_file": DEFAULT_MASTER_CSV_NAME,
        "custom_log_file": custom_log_rel,
        "inspect_log_file": inspect_rel_path,
        "varied_log_file": varied_log_rel_path or ""
    }

    # 1. Save detailed JSON to custom_logs/
    save_custom_log(
        custom_logs_dir=CUSTOM_LOGS_DIR,
        file_prefix=task_name,
        meta=meta,
        results=live_sample_results,
        varied_response_summary=varied_response_summary,
        timestamp_slug=ts_slug
    )

    # 2. Append summary row to master CSV
    master_csv_path = EXPERIMENT_A2_DIR / DEFAULT_MASTER_CSV_NAME
    append_to_master_csv(
        master_csv_path=master_csv_path,
        meta=meta,
        results=live_sample_results,
        custom_log_rel_path=custom_log_rel,
        inspect_log_rel_path=inspect_rel_path,
        varied_log_rel_path=varied_log_rel_path or ""
    )

    print(f"\n[AUTO-SYNC] Evaluation ID     : {eval_id}")
    print(f"[AUTO-SYNC] Detailed JSON Log : {custom_log_rel}")
    print(f"[AUTO-SYNC] Master CSV Row    : {DEFAULT_MASTER_CSV_NAME}")
    if inspect_rel_path:
        print(f"[AUTO-SYNC] Inspect AI Log  : {inspect_rel_path}")

    print(f">>> [RUN {run_index}] Completed! Successfully collected {len(scores_by_id)} scores.")
    run_record = {
        "eval_id": eval_id,
        "run_label": run_label,
        "run_index": run_index,
        "custom_log_path": custom_log_path,
        "custom_log_rel": custom_log_rel,
        "inspect_log_rel": inspect_rel_path,
        "meta": meta,
        "results": live_sample_results
    }
    return scores_by_id, run_record


# ==============================================================================
# 4. CSV SESSION FILE MANAGEMENT
# ==============================================================================
def save_varied_results_csv(
    csv_path: Path,
    varied_data: List[Dict[str, Any]],
    run_columns: List[str]
) -> None:
    """
    Saves or updates the session CSV file in varied_response_logs/.
    Columns: prompt_id, category, prompt, run_1, run_2, ...
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt_id", "category", "prompt"] + run_columns

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in varied_data:
            row_dict = {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "prompt": row["prompt"]
            }
            for col in run_columns:
                row_dict[col] = row.get(col, "")
            writer.writerow(row_dict)

    print(f"[CSV SAVED] Successfully written to: {csv_path}")


def print_summary_table(
    varied_data: List[Dict[str, Any]],
    run_columns: List[str]
) -> None:
    """Prints a clean ASCII table of all varied prompts and their scores across runs."""
    if not varied_data:
        print("\nNo prompts to display in summary table.")
        return

    print("\n" + "=" * 105)
    print(f"PROMPTS SUMMARY ({len(varied_data)} Prompts Tracked Across {len(run_columns)} Runs)")
    print("=" * 105)

    id_w = 16
    cat_w = 20
    prompt_w = 40
    run_w = 7

    headers = [f"{'PROMPT ID':<{id_w}}", f"{'CATEGORY':<{cat_w}}", f"{'PROMPT (SNIPPET)':<{prompt_w}}"]
    for col in run_columns:
        headers.append(f"{col:>{run_w}}")
    print(" | ".join(headers))
    print("-" * 105)

    for item in varied_data:
        pid = item["prompt_id"][:id_w]
        cat = item["category"][:cat_w]
        p_text = item["prompt"].replace("\n", " ").strip()
        if len(p_text) > prompt_w:
            p_text = p_text[:prompt_w - 3] + "..."

        row_fields = [f"{pid:<{id_w}}", f"{cat:<{cat_w}}", f"{p_text:<{prompt_w}}"]
        for col in run_columns:
            score = item.get(col, "-")
            row_fields.append(f"{str(score):>{run_w}}")
        print(" | ".join(row_fields))

    print("=" * 105 + "\n")


# ==============================================================================
# 5. MAIN WORKFLOW: BASELINE RUNS -> VARIANCE DETECTION -> INTERACTIVE RE-RUNS
# ==============================================================================
def run_serial_batch(
    padding: str = DEFAULT_PADDING,
    position: str = DEFAULT_POSITION,
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_JUDGE,
    initial_runs: int = DEFAULT_INITIAL_RUNS,
    additional_runs: Optional[int] = DEFAULT_ADDITIONAL_RUNS,
    limit: Optional[int] = DEFAULT_LIMIT,
    max_connections: int = MAX_CONNECTIONS
) -> None:
    norm_pad = normalize_padding(padding)
    norm_pos = normalize_position(position)

    print("\n" + "=" * 80)
    print("SERIAL BATCH EVALUATION WORKFLOW - VARIANCE DETECTOR & ITERATIVE RUNNER")
    print("=" * 80)
    print(f"Padding Size     : {norm_pad}")
    print(f"Prompt Position  : {norm_pos}")
    print(f"Target Model     : {model}")
    print(f"Judge Model      : {judge_model}")
    print(f"Initial Baseline : {initial_runs} runs across full dataset")
    print(f"Sample Limit     : {limit if limit else 'Full 100 Prompts'}")
    print(f"Max Connections  : {max_connections} ({'Strictly Serial' if max_connections == 1 else 'Parallel'})")
    print(f"Inspect Logs     : {INSPECT_LOGS_DIR}")
    print(f"Varied Logs Dir  : {VARIED_LOGS_DIR}")
    print("=" * 80 + "\n")

    # 1. Load initial full dataset
    all_dataset_items = load_full_dataset(padding=norm_pad, position=norm_pos, limit=limit)
    total_samples = len(all_dataset_items)
    print(f"[DATASET] Loaded {total_samples} prompts for baseline evaluation.")

    # 2. Phase 1: Execute Initial Baseline Runs
    run_columns: List[str] = []
    # prompt_id -> {col_name: score}
    scores_matrix: Dict[str, Dict[str, int]] = {item["id"]: {} for item in all_dataset_items}
    completed_run_records: Dict[str, Dict[str, Any]] = {}

    for run_idx in range(1, initial_runs + 1):
        col_name = f"{RUN_COLUMN_PREFIX}{run_idx}"
        run_columns.append(col_name)

        run_label = f"run_{run_idx}"
        print(f"\n=======================================================")
        print(f"STARTING BASELINE RUN {run_idx} OF {initial_runs} ({col_name})")
        print(f"=======================================================")

        run_scores, run_record = execute_inspect_run(
            items_to_evaluate=all_dataset_items,
            run_label=run_label,
            run_index=run_idx,
            padding=norm_pad,
            position=norm_pos,
            model=model,
            judge_model=judge_model,
            max_connections=max_connections,
            log_dir=INSPECT_LOGS_DIR
        )

        completed_run_records[col_name] = run_record

        for item in all_dataset_items:
            pid = item["id"]
            scores_matrix[pid][col_name] = run_scores.get(pid, -1)

    # 3. Phase 2: Variance Detection
    print("\n" + "=" * 80)
    print("ANALYZING SCORE VARIANCE ACROSS BASELINE RUNS...")
    print("=" * 80)

    varied_items: List[Dict[str, Any]] = []
    consistent_count = 0

    for item in all_dataset_items:
        pid = item["id"]
        run_scores = [scores_matrix[pid].get(c, -1) for c in run_columns]
        # A question varied if it showed even a single different score across runs
        has_variance = len(set(run_scores)) > 1

        if has_variance:
            row_data = {
                "prompt_id": pid,
                "category": item["category"],
                "prompt": item["raw_prompt"],
                "item_ref": item
            }
            for c in run_columns:
                row_data[c] = scores_matrix[pid].get(c, -1)
            varied_items.append(row_data)
        else:
            consistent_count += 1

    print(f"Total Prompts Evaluated : {total_samples}")
    print(f"Consistent Prompts      : {consistent_count} (identical score in all {initial_runs} runs)")
    print(f"Varied Prompts Found    : {len(varied_items)} (showed score differences)")

    # Prepare CSV file path for this execution session
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"varied_eval_{norm_pad}_{norm_pos}_{timestamp_slug}.csv"
    csv_path = VARIED_LOGS_DIR / csv_filename
    varied_log_rel = f"varied_response_logs/{csv_filename}"

    varied_summary = {
        "session_csv": varied_log_rel,
        "total_evaluated": total_samples,
        "consistent_prompts_count": consistent_count,
        "varied_prompts_count": len(varied_items),
        "has_variance": len(varied_items) > 0,
        "varied_prompts": [
            {
                "prompt_id": v["prompt_id"],
                "category": v["category"],
                "prompt": v["prompt"],
                "scores": {c: scores_matrix[v["prompt_id"]].get(c, -1) for c in run_columns}
            }
            for v in varied_items
        ]
    }

    # Update previous baseline custom logs with varied_responses section and varied_log_file reference
    baseline_eval_ids = set()
    for col_name, rec in completed_run_records.items():
        baseline_eval_ids.add(rec["eval_id"])
        custom_p = rec["custom_log_path"]
        if custom_p.exists():
            try:
                with open(custom_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["metadata"]["varied_log_file"] = varied_log_rel
                data["varied_responses"] = varied_summary
                with open(custom_p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Warning] Failed to update custom log {custom_p.name} with varied response: {e}")

    # Also update master CSV entries for baseline runs to include varied_log_file
    update_master_csv_varied_log(
        master_csv_path=EXPERIMENT_A2_DIR / DEFAULT_MASTER_CSV_NAME,
        target_eval_ids=baseline_eval_ids,
        varied_log_rel_path=varied_log_rel
    )

    # If no variance was found among the samples
    if not varied_items:
        print("\n[INFO] No prompts exhibited score variance across the initial baseline runs.")
        print(f"[CSV SAVED] Writing all {total_samples} baseline prompts to CSV for reference.")
        baseline_rows = []
        for item in all_dataset_items:
            row = {
                "prompt_id": item["id"],
                "category": item["category"],
                "prompt": item["raw_prompt"],
                "item_ref": item
            }
            for c in run_columns:
                row[c] = scores_matrix[item["id"]].get(c, -1)
            baseline_rows.append(row)
        save_varied_results_csv(csv_path, baseline_rows, run_columns)
        print_summary_table(baseline_rows, run_columns)

        # Allow user to continue running more evals if desired
        target_eval_items = all_dataset_items
        active_rows = baseline_rows
    else:
        # Save varied prompts CSV and display summary
        print_summary_table(varied_items, run_columns)
        save_varied_results_csv(csv_path, varied_items, run_columns)
        target_eval_items = [v["item_ref"] for v in varied_items]
        active_rows = varied_items

    # 4. Phase 3 & 4: Interactive / Configured Additional Runs Loop
    current_run_count = initial_runs
    pending_additional_runs = additional_runs

    while True:
        num_to_run: int = 0

        if pending_additional_runs is not None:
            num_to_run = pending_additional_runs
            pending_additional_runs = None  # Consume CLI argument once
        else:
            print("\n" + "-" * 75)
            prompt_msg = (
                f"Would you like to run another set of evals on these {len(active_rows)} prompts?\n"
                f"Enter number of additional runs to execute (or press Enter / 0 / 'q' to finish): "
            )
            try:
                user_input = input(prompt_msg).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[EXIT] User ended input.")
                break

            if not user_input or user_input.lower() in ("0", "q", "exit", "n", "no"):
                print("[EXIT] Evaluation session concluded.")
                break

            try:
                num_to_run = int(user_input)
                if num_to_run <= 0:
                    print("[EXIT] Evaluation session concluded.")
                    break
            except ValueError:
                print(f"[Warning] Invalid input '{user_input}'. Please enter a valid positive integer.")
                continue

        # Execute additional runs on the target dataset
        print(f"\n>>> Executing {num_to_run} additional run(s) on {len(active_rows)} prompts...")

        for _ in range(num_to_run):
            current_run_count += 1
            new_col = f"{RUN_COLUMN_PREFIX}{current_run_count}"
            run_columns.append(new_col)

            run_label = f"varied_run_{current_run_count}"
            print(f"\n=======================================================")
            print(f"RUNNING EVALUATION #{current_run_count} ({new_col})")
            print(f"Targeting: {len(active_rows)} prompts")
            print(f"=======================================================")

            new_scores, new_record = execute_inspect_run(
                items_to_evaluate=target_eval_items,
                run_label=run_label,
                run_index=current_run_count,
                padding=norm_pad,
                position=norm_pos,
                model=model,
                judge_model=judge_model,
                max_connections=max_connections,
                log_dir=INSPECT_LOGS_DIR,
                varied_log_rel_path=varied_log_rel,
                varied_response_summary=varied_summary
            )

            # Update scores for each row
            for row in active_rows:
                pid = row["prompt_id"]
                row[new_col] = new_scores.get(pid, -1)
                if pid in scores_matrix:
                    scores_matrix[pid][new_col] = new_scores.get(pid, -1)

            # Update varied_summary scores with the new run
            for vp in varied_summary["varied_prompts"]:
                vp_id = vp["prompt_id"]
                if vp_id in scores_matrix:
                    vp["scores"][new_col] = scores_matrix[vp_id].get(new_col, -1)

            # Re-save the CSV immediately after each run completes
            save_varied_results_csv(csv_path, active_rows, run_columns)

        # Print updated summary table
        print_summary_table(active_rows, run_columns)
        print(f"[UPDATED] Session CSV updated with {len(run_columns)} total run columns at:\n  {csv_path}")

    # Final wrap-up report
    print("\n" + "=" * 80)
    print("SERIAL BATCH EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total Runs Recorded   : {len(run_columns)} ({', '.join(run_columns)})")
    print(f"Varied Prompts Count  : {len(varied_items)}")
    print(f"Final CSV Location    : {csv_path}")
    print(f"Inspect Visual Logs   : inspect view --log-dir \"{INSPECT_LOGS_DIR}\"")
    print("=" * 80 + "\n")


# ==============================================================================
# 6. CLI ARGUMENT PARSER
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Serial batch evaluation runner to detect prompt score variance across runs."
    )
    parser.add_argument(
        "--padding", "-p",
        type=str,
        default=DEFAULT_PADDING,
        choices=["none", "0k", "1k", "2k", "5k", "10k", "20k", "50k"],
        help=f"Context padding size (default: {DEFAULT_PADDING})"
    )
    parser.add_argument(
        "--position", "--pos",
        type=str,
        default=DEFAULT_POSITION,
        choices=["start", "middle", "end"],
        help=f"Harmful prompt position within padding (default: {DEFAULT_POSITION})"
    )
    parser.add_argument(
        "--initial-runs", "-r",
        type=int,
        default=DEFAULT_INITIAL_RUNS,
        help=f"Number of initial baseline runs across all prompts (default: {DEFAULT_INITIAL_RUNS})"
    )
    parser.add_argument(
        "--additional-runs", "-a",
        type=int,
        default=DEFAULT_ADDITIONAL_RUNS,
        help="Number of additional runs to execute on the varied dataset (default: interactive CLI prompt)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Target Gemini model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--judge-model", "-j",
        type=str,
        default=DEFAULT_JUDGE,
        help=f"Safety judge model (default: {DEFAULT_JUDGE})"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_LIMIT,
        help="Optional limit on number of prompts to evaluate (e.g. 5 for testing)"
    )
    parser.add_argument(
        "--max-connections", "--concurrency",
        type=int,
        default=MAX_CONNECTIONS,
        help=f"Maximum concurrent connections to model (default: {MAX_CONNECTIONS} for serial execution)"
    )
    parser.add_argument(
        "--printAPI", "--print-api", "--printapi",
        dest="print_api",
        action="store_true",
        help="Print active API key configuration status."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.print_api:
        print_available_api_keys()
        sys.exit(0)

    try:
        run_serial_batch(
            padding=args.padding,
            position=args.position,
            model=args.model,
            judge_model=args.judge_model,
            initial_runs=args.initial_runs,
            additional_runs=args.additional_runs,
            limit=args.limit,
            max_connections=args.max_connections
        )
    except QuotaExceededError as qe:
        print(f"\n[QUOTA ERROR] {qe}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ABORTED] Evaluation interrupted by user.")
        sys.exit(0)
