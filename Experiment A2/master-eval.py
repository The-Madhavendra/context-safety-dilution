"""
master-eval.py
==============
Master orchestration pipeline for the Experiment A2 benchmark:
- 5 Context Sizes: 1k, 5k, 10k, 25k, 50k
- 3 Positions    : start, end, middle
- 1 Iteration    : single iteration per context-position combo
- Total Runs     : 5 x 3 = 15 runs
- Total Samples  : 15 runs x 100 samples = 1,500 samples
- Total Requests : 1,500 x 2 = 3,000 API requests (1 target model + 1 judge model)

Key Features:
1. Environment Diagnostics: Validates API configuration before runs and provides clear telemetry.
2. Pacing with Random Float Timeout: 1-6 sec random timeout between each sample run.
3. Bulletproof Failure Logger: Records unrecoverable sample errors to `fail_logger.json`
   without halting the entire evaluation batch, ready for downstream repair via `fail-repair.py`.
4. Intelligent Resume: Checks `master_eval_results.csv` and automatically skips already-completed
   runs so interruptions never waste quota.
5. Real-time Progress Tracking: Live terminal telemetry for runs, samples, and quota health.
6. Unified Persistence: Saves detailed JSON logs to `custom_logs/` and summary rows to `master_eval_results.csv`.
"""

import os
import sys
import time
import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load/reload .env with override
try:
    from dotenv import load_dotenv
    for parent in [Path(__file__).resolve().parent, PROJECT_ROOT]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
except ImportError:
    pass

# Import core evaluation functions and constants
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
    call_gemini,
    call_judge,
    format_summary_table,
    save_custom_log,
    append_to_master_csv,
    generate_eval_id,
    get_rolling_tpm_guard
)

from Utils.api_management import (
    discover_api_keys,
    get_gemini_manager,
    print_available_keys,
    QuotaExceededError
)

# ==============================================================================
# CONSTANTS & DEFAULTS FOR THE 15-RUN EXPERIMENT
# ==============================================================================
BENCHMARK_CONTEXT_SIZES: List[str] = ["1k", "5k", "10k", "25k", "50k"]
BENCHMARK_POSITIONS: List[str] = ["start", "end", "middle"]
DEFAULT_ITERATIONS: int = 1

TARGET_MODEL: str = DEFAULT_GEMINI_MODEL       # google/gemini-3.5-flash-lite
JUDGE_MODEL: str = DEFAULT_JUDGE_MODEL         # google/gemini-3.1-flash-lite

FAIL_LOGGER_PATH: Path = EXPERIMENT_A2_DIR / "fail_logger.json"
MASTER_CSV_PATH: Path = EXPERIMENT_A2_DIR / DEFAULT_MASTER_CSV_NAME


# ==============================================================================
# 1. DYNAMIC API KEY REFRESH & DIAGNOSTICS
# ==============================================================================
def refresh_and_display_api_keys() -> Dict[str, Any]:
    """
    Validates API key configuration and displays diagnostic status for
    both the target model and the evaluation judge model.
    """
    print("\n" + "=" * 78)
    print(" VERIFYING API CONFIGURATION & CONNECTION STATUS")
    print("=" * 78)

    # 1. Reload .env file if present
    for p in [EXPERIMENT_A2_DIR, PROJECT_ROOT]:
        dot_env = p / ".env"
        if dot_env.exists():
            try:
                load_dotenv(dot_env, override=True)
                print(f"[Config] Loaded environment variables from: {dot_env.name}")
            except Exception as e:
                print(f"[Warning] Failed to reload {dot_env.name}: {e}")

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    masked_google = f"{google_key[:6]}...{google_key[-4:]}" if google_key and len(google_key) >= 12 else ("CONFIGURED" if google_key else "NOT SET")
    masked_groq = f"{groq_key[:6]}...{groq_key[-4:]}" if groq_key and len(groq_key) >= 12 else ("CONFIGURED" if groq_key else "NOT SET")

    target_clean = TARGET_MODEL.replace("google/", "").strip()
    judge_clean = JUDGE_MODEL.replace("google/", "").strip()

    t_status = 'READY' if google_key else 'MISSING KEY'
    j_status = 'READY' if google_key else 'MISSING KEY'

    print(f" Target Model ({target_clean:<22}) : {t_status}")
    print(f" Judge Model  ({judge_clean:<22}) : {j_status}")
    print(f" Google API Key                      : {masked_google}")
    print(f" Groq API Key                        : {masked_groq}")
    print("=" * 78 + "\n")

    return {
        "google_configured": bool(google_key),
        "groq_configured": bool(groq_key)
    }


# ==============================================================================
# 2. FAIL LOGGER (fail_logger.json)
# ==============================================================================
def record_failure_to_logger(
    fail_logger_path: Path,
    failure_entry: Dict[str, Any]
) -> None:
    """
    Appends a failed sample record to fail_logger.json in an atomic, safe manner.
    Records all context required for downstream replay via fail-repair.py:
    eval_id, run_label, context_size, position, sample_id, stage, raw prompt,
    padded prompt, and exact error details.
    """
    try:
        fail_logger_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_failures": 0,
            "failures": []
        }

        if fail_logger_path.exists():
            try:
                with open(fail_logger_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "failures" in loaded:
                        data = loaded
            except Exception:
                pass

        if "failures" not in data or not isinstance(data["failures"], list):
            data["failures"] = []

        data["failures"].append(failure_entry)
        data["total_failures"] = len(data["failures"])
        data["updated_at"] = datetime.now().isoformat()

        # Atomic write via temp file
        temp_file = fail_logger_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(fail_logger_path)

        print(f"[FAIL LOGGER] Recorded failure for prompt '{failure_entry.get('prompt_id')}' to {fail_logger_path.name}")
    except Exception as e:
        print(f"[Warning] Failed to write to {fail_logger_path.name}: {e}")


# ==============================================================================
# 3. COMPLETED RUNS DETECTOR (RESUME SUPPORT)
# ==============================================================================
def get_completed_runs(master_csv_path: Path, min_samples: int = 100) -> Set[str]:
    """
    Reads master_eval_results.csv and returns a set of completed run signatures:
    "{context_size}_{position}" that have at least min_samples evaluated.
    """
    completed: Set[str] = set()
    if not master_csv_path.exists():
        return completed

    try:
        import csv
        with open(master_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ctx = str(row.get("context_size", "")).strip().lower()
                pos = str(row.get("harmful_prompt_position", "")).strip().lower()
                try:
                    count = int(row.get("total_prompts_evaluated", 0))
                except (ValueError, TypeError):
                    count = 0

                if ctx and pos and count >= min_samples:
                    sig = f"{ctx}_{pos}"
                    completed.add(sig)
    except Exception as e:
        print(f"[Warning] Could not inspect master CSV for completed runs: {e}")

    return completed



# ==============================================================================
# 4. SINGLE RUN EXECUTION ENGINE
# ==============================================================================
def execute_single_run(
    run_idx: int,
    total_runs: int,
    context_size: str,
    position: str,
    iteration: int,
    master_prompt_template: str,
    limit: Optional[int] = None,
    pacing_delay: float = 1.0,
    fail_logger_path: Path = FAIL_LOGGER_PATH,
    master_csv_path: Path = MASTER_CSV_PATH
) -> Dict[str, Any]:
    """
    Executes a single 100-sample run for the given (context_size, position, iteration).
    - Assembles padded dataset using 10x10 mapping
    - Calls target model (gemini-3.5-flash-lite)
    - Calls judge model (gemini-3.1-flash-lite)
    - Logs unrecoverable sample errors to fail_logger.json
    - Saves run results to custom_logs/
    - Appends row to master_eval_results.csv
    """
    run_label = f"run_{iteration}"
    task_name = f"A2_{context_size}_{position}_{run_label}"
    eval_id = generate_eval_id(context_size, position, run_label)

    print("\n" + "=" * 78)
    print(f" [RUN {run_idx:02d}/{total_runs:02d}] {task_name.upper()}")
    print("=" * 78)
    print(f" Context Size : {context_size}")
    print(f" Position     : {position}")
    print(f" Iteration    : {iteration} ({run_label})")
    print(f" Target Model : {TARGET_MODEL}")
    print(f" Safety Judge : {JUDGE_MODEL}")
    print(f" Evaluation ID: {eval_id}")
    print("=" * 78)

    # 1. Build mapped dataset
    dataset = build_mapped_dataset(
        harmful_prompts_file=BASE_HARMFUL_PROMPTS_PATH,
        contexts_chunked_file=CONTEXTS_CHUNKED_PATH,
        padding=context_size,
        position=position
    )

    if limit is not None and limit < len(dataset):
        dataset = dataset[:limit]
        print(f"[Notice] Sample limit applied: evaluating {limit} samples.")

    total_samples = len(dataset)
    results: List[Dict[str, Any]] = []
    failed_samples_count = 0
    start_time = time.time()

    for idx, item in enumerate(dataset, start=1):
        prompt_id = item["id"]
        category = item["category"]
        raw_prompt = item["raw_prompt"]
        padded_prompt = item["padded_prompt"]
        context_category = item["context_category"]

        print(f"[{run_idx:02d}/{total_runs:02d}] [{idx:03d}/{total_samples:03d}] {prompt_id:<12} ({category}) | Target Call...", end="", flush=True)

        # Individual Sample Rolling TPM Guard (for 50k context: max 4 samples per rolling 60s)
        if str(context_size).lower().strip() == "50k":
            get_rolling_tpm_guard().check_and_wait(
                padding_size="50k",
                sample_id=prompt_id
            )

        # ----------------------------------------------------------------------
        # STEP A: Target Model Call
        # ----------------------------------------------------------------------
        gemini_response = ""
        target_error: Optional[Exception] = None

        try:
            gemini_response = call_gemini(
                prompt=padded_prompt,
                model_name=TARGET_MODEL
            )
            print(" Done | Judge Call...", end="", flush=True)
        except Exception as e:
            target_error = e
            failed_samples_count += 1
            print(f" FAILED ({type(e).__name__})", flush=True)

            # Record failure to fail_logger.json
            fail_record = {
                "fail_id": f"fail_{eval_id}_{prompt_id}_target",
                "eval_id": eval_id,
                "run_label": run_label,
                "context_size": context_size,
                "position": position,
                "iteration": iteration,
                "sample_index": idx,
                "prompt_id": prompt_id,
                "category": category,
                "context_category": context_category,
                "raw_prompt": raw_prompt,
                "padded_prompt": padded_prompt,
                "target_model": TARGET_MODEL,
                "judge_model": JUDGE_MODEL,
                "stage": "target_model",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            record_failure_to_logger(fail_logger_path, fail_record)

            results.append({
                "id": prompt_id,
                "category": category,
                "context_category": context_category,
                "padding": context_size,
                "position": position,
                "raw_prompt": raw_prompt,
                "gemini_response": f"[ERROR] Target Model Failed: {e}",
                "judge_score": -1,
                "judge_reason": f"Target model failed: {e}",
                "judge_evidence": "",
                "status": "failed_target"
            })

            # Check if API rate limit or quota exceeded
            if isinstance(e, QuotaExceededError) or "QUOTA EXHAUSTED" in str(e):
                print(f"\n[ALERT] API rate limit or quota exceeded for target model!")
                print(f"[ACTION REQUIRED] Please verify your GOOGLE_API_KEY in your environment or .env file.")
                # We stop the sample loop but preserve what was completed
                break

            sample_timeout = round(random.uniform(3.5, 6.0), 2)
            time.sleep(sample_timeout)
            continue

        # Inter-request delay between target response and safety judge (to pace single shared API key)
        post_target_delay = round(random.uniform(3.5, 6.0), 2)
        time.sleep(post_target_delay)

        # ----------------------------------------------------------------------
        # STEP B: Judge Model Call
        # ----------------------------------------------------------------------
        judge_res: Dict[str, Any] = {}
        judge_error: Optional[Exception] = None

        try:
            judge_res = call_judge(
                harmful_request=raw_prompt,
                model_response=gemini_response,
                master_prompt_template=master_prompt_template,
                judge_model=JUDGE_MODEL
            )
            score_val = judge_res.get("score", -1)
            print(f" Score: {score_val}/4", flush=True)

            # Check if judge encountered an error response
            if score_val == -1 and "error" in str(judge_res.get("reason", "")).lower():
                failed_samples_count += 1
                fail_record = {
                    "fail_id": f"fail_{eval_id}_{prompt_id}_judge",
                    "eval_id": eval_id,
                    "run_label": run_label,
                    "context_size": context_size,
                    "position": position,
                    "iteration": iteration,
                    "sample_index": idx,
                    "prompt_id": prompt_id,
                    "category": category,
                    "context_category": context_category,
                    "raw_prompt": raw_prompt,
                    "padded_prompt": padded_prompt,
                    "gemini_response": gemini_response,
                    "target_model": TARGET_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "stage": "judge_model",
                    "error_type": "JudgeEvaluationError",
                    "error_message": judge_res.get("reason", "Judge returned score -1"),
                    "timestamp": datetime.now().isoformat()
                }
                record_failure_to_logger(fail_logger_path, fail_record)

        except Exception as e:
            judge_error = e
            failed_samples_count += 1
            print(f" FAILED ({type(e).__name__})", flush=True)

            fail_record = {
                "fail_id": f"fail_{eval_id}_{prompt_id}_judge",
                "eval_id": eval_id,
                "run_label": run_label,
                "context_size": context_size,
                "position": position,
                "iteration": iteration,
                "sample_index": idx,
                "prompt_id": prompt_id,
                "category": category,
                "context_category": context_category,
                "raw_prompt": raw_prompt,
                "padded_prompt": padded_prompt,
                "gemini_response": gemini_response,
                "target_model": TARGET_MODEL,
                "judge_model": JUDGE_MODEL,
                "stage": "judge_model",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            record_failure_to_logger(fail_logger_path, fail_record)
            judge_res = {
                "score": -1,
                "reason": f"Judge call error: {e}",
                "evidence": ""
            }

            if isinstance(e, QuotaExceededError) or "QUOTA EXHAUSTED" in str(e):
                print(f"\n[ALERT] API rate limit or quota exceeded for judge model!")
                break

        results.append({
            "id": prompt_id,
            "category": category,
            "context_category": context_category,
            "padding": context_size,
            "position": position,
            "raw_prompt": raw_prompt,
            "gemini_response": gemini_response,
            "judge_score": judge_res.get("score", -1),
            "judge_reason": judge_res.get("reason", ""),
            "judge_evidence": judge_res.get("evidence", ""),
            "status": "success" if judge_res.get("score", -1) >= 0 else "failed_judge"
        })

        # Progressive save: immediately commit progress to custom_logs and master_eval_results.csv
        try:
            CUSTOM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            sample_meta: Dict[str, Any] = {
                "eval_id": eval_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_name": task_name,
                "run_label": run_label,
                "padding": context_size,
                "position": position,
                "iteration": iteration,
                "gemini_model": TARGET_MODEL,
                "judge_model": JUDGE_MODEL,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "total_evaluated": len(results),
                "failed_samples": failed_samples_count,
                "master_csv_file": master_csv_path.name
            }
            c_path = save_custom_log(
                custom_logs_dir=CUSTOM_LOGS_DIR,
                file_prefix=task_name,
                meta=sample_meta,
                results=results
            )
            c_rel = f"custom_logs/{c_path.name}"
            sample_meta["custom_log_file"] = c_rel
            append_to_master_csv(
                master_csv_path=master_csv_path,
                meta=sample_meta,
                results=results,
                custom_log_rel_path=c_rel,
                inspect_log_rel_path="",
                varied_log_rel_path=""
            )
        except Exception as se:
            print(f"\n[Warning] Progressive sample save error: {se}")

        # Inter-request delay after safety judge grading (to pace single shared API key)
        sample_timeout = round(random.uniform(3.5, 6.0), 2)
        time.sleep(sample_timeout)

    elapsed_seconds = time.time() - start_time
    results.sort(key=lambda x: x["id"])

    # --------------------------------------------------------------------------
    # STEP C: Save Detailed Custom Log JSON
    # --------------------------------------------------------------------------
    meta: Dict[str, Any] = {
        "eval_id": eval_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": task_name,
        "run_label": run_label,
        "padding": context_size,
        "position": position,
        "gemini_model": TARGET_MODEL,
        "judge_model": JUDGE_MODEL,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "total_evaluated": len(results),
        "failed_samples": failed_samples_count,
        "master_csv_file": master_csv_path.name
    }

    CUSTOM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    custom_log_path = save_custom_log(
        custom_logs_dir=CUSTOM_LOGS_DIR,
        file_prefix=task_name,
        meta=meta,
        results=results
    )
    custom_log_rel = f"custom_logs/{custom_log_path.name}"
    meta["custom_log_file"] = custom_log_rel

    # --------------------------------------------------------------------------
    # STEP D: Append Summary Row to master_eval_results.csv
    # --------------------------------------------------------------------------
    append_to_master_csv(
        master_csv_path=master_csv_path,
        meta=meta,
        results=results,
        custom_log_rel_path=custom_log_rel,
        inspect_log_rel_path="",
        varied_log_rel_path=""
    )

    # --------------------------------------------------------------------------
    # STEP E: Summary Display
    # --------------------------------------------------------------------------
    summary_table = format_summary_table(results, meta)
    print("\n" + "=" * 78)
    print(f" [RUN {run_idx:02d}/{total_runs:02d} COMPLETED] {task_name}")
    print("=" * 78)
    print(summary_table)
    print(f" Detailed JSON Log : {custom_log_rel}")
    print(f" Master CSV File   : {master_csv_path.name}")
    if failed_samples_count > 0:
        print(f" Failures Logged   : {failed_samples_count} sample(s) written to {fail_logger_path.name}")
    print("=" * 78 + "\n")

    return {
        "task_name": task_name,
        "eval_id": eval_id,
        "total_evaluated": len(results),
        "failed_samples": failed_samples_count,
        "custom_log_path": custom_log_path
    }


def execute_single_run_inspect_process(
    run_idx: int,
    total_runs: int,
    context_size: str,
    position: str,
    iteration: int,
    limit: Optional[int] = None,
    fail_logger_path: Path = FAIL_LOGGER_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    force_rerun: bool = False
) -> Dict[str, Any]:
    """
    Executes a single evaluation run as a dedicated Inspect AI child process.
    - Spawns python Utils/inspect_eval.py with exact run parameters
    - Automatically generates official Inspect .eval logs in Experiment A2/logs/ (viewable via inspect view)
    - Records detailed JSON to custom_logs/ and appends row to master_eval_results.csv
    - Terminates cleanly upon completion, freeing all OS and event loop resources
    - If non-zero exit code occurs, captures failure and records to fail_logger.json
    """
    run_label = f"run_{iteration}"
    task_name = f"A2_{context_size}_{position}_{run_label}"
    eval_id = generate_eval_id(context_size, position, run_label)

    print("\n" + "=" * 78)
    print(f" [RUN {run_idx:02d}/{total_runs:02d}] SPAWNING INSPECT AI CHILD PROCESS: {task_name.upper()}")
    print("=" * 78)
    print(f" Context Size : {context_size}")
    print(f" Position     : {position}")
    print(f" Iteration    : {iteration} ({run_label})")
    print(f" Target Model : {TARGET_MODEL}")
    print(f" Safety Judge : {JUDGE_MODEL}")
    print(f" Evaluation ID: {eval_id}")
    print("=" * 78)

    inspect_script = PROJECT_ROOT / "Utils" / "inspect_eval.py"
    inspect_task_target = f"{inspect_script}@a2_context_padding_eval"
    cmd = [
        sys.executable,
        "-m", "inspect_ai", "eval",
        inspect_task_target,
        "--model", TARGET_MODEL,
        "-T", f"padding={context_size}",
        "-T", f"position={position}",
        "-T", f"iteration={iteration}",
        "-T", f"judge_model={JUDGE_MODEL}",
        "--max-samples", "1",
        "--max-connections", "1",
        "--adaptive-connections", "false",
        "--no-fail-on-error",
        "--display", "plain"
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if force_rerun:
        cmd.extend(["-T", "force=True"])

    # Ensure PYTHONPATH includes project root for child process
    child_env = os.environ.copy()
    if "PYTHONPATH" in child_env:
        child_env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{child_env['PYTHONPATH']}"
    else:
        child_env["PYTHONPATH"] = str(PROJECT_ROOT)

    start_time = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=child_env)
        elapsed = time.time() - start_time

        if proc.returncode in (130, 2, -2):
            print("\n" + "=" * 78)
            print(f"[KEYBOARD INTERRUPT] Evaluation interrupted by user during '{task_name}'.")
            print("All completed sample entries and counts have been saved to master_eval_results.csv.")
            print("Re-running master-eval.py will seamlessly resume from the next uncompleted sample.")
            print("=" * 78 + "\n")
            sys.exit(130)

        success = (proc.returncode == 0)

        if not success:
            print(f"\n[ALERT] Inspect AI child process exited with return code: {proc.returncode}")
            fail_record = {
                "fail_id": f"fail_{eval_id}_process_error",
                "eval_id": eval_id,
                "run_label": run_label,
                "context_size": context_size,
                "position": position,
                "iteration": iteration,
                "target_model": TARGET_MODEL,
                "judge_model": JUDGE_MODEL,
                "stage": "inspect_child_process",
                "error_type": "ProcessExitCodeError",
                "error_message": f"inspect_eval.py exited with return code {proc.returncode}",
                "timestamp": datetime.now().isoformat()
            }
            record_failure_to_logger(fail_logger_path, fail_record)
            return {
                "task_name": task_name,
                "eval_id": eval_id,
                "failed_samples": 1,
                "returncode": proc.returncode,
                "elapsed": elapsed
            }
        else:
            print(f"\n[SUCCESS] Run '{task_name}' finished in {elapsed:.1f}s. Log saved to Inspect AI logs/.")
            return {
                "task_name": task_name,
                "eval_id": eval_id,
                "failed_samples": 0,
                "returncode": 0,
                "elapsed": elapsed
            }

    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print(f"[KEYBOARD INTERRUPT] Master execution interrupted by user during '{task_name}'.")
        print("All completed sample entries and counts have been saved to master_eval_results.csv.")
        print("Re-running master-eval.py will seamlessly resume from the next uncompleted sample.")
        print("=" * 78 + "\n")
        sys.exit(130)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[EXCEPTION] Failed to launch Inspect child process: {e}")
        fail_record = {
            "fail_id": f"fail_{eval_id}_process_launch_exception",
            "eval_id": eval_id,
            "run_label": run_label,
            "context_size": context_size,
            "position": position,
            "iteration": iteration,
            "target_model": TARGET_MODEL,
            "judge_model": JUDGE_MODEL,
            "stage": "child_process_launch",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        record_failure_to_logger(fail_logger_path, fail_record)
        return {
            "task_name": task_name,
            "eval_id": eval_id,
            "failed_samples": 1,
            "returncode": -1,
            "elapsed": elapsed
        }


# ==============================================================================
# 5. MASTER ORCHESTRATION PIPELINE
# ==============================================================================
def run_master_evaluation_suite(
    context_sizes: List[str] = BENCHMARK_CONTEXT_SIZES,
    positions: List[str] = BENCHMARK_POSITIONS,
    iterations: int = DEFAULT_ITERATIONS,
    limit: Optional[int] = None,
    max_runs: Optional[int] = None,
    runner: str = "inspect",
    pacing_delay: float = 2.0,
    force_rerun: bool = False,
    dry_run: bool = False
) -> None:
    """
    Executes the 15-run evaluation matrix:
    5 context sizes x 3 positions x 1 iteration = 15 runs
    Total: 1,500 samples / 3,000 API calls
    Runner mode:
      'inspect': Spawns dedicated child process per run via inspect_eval.py (viewable in inspect view)
      'direct' : In-process sequential runner
    """
    start_suite_time = time.time()

    # Step 1: Refresh keys and quotas
    key_diagnostics = refresh_and_display_api_keys()

    # Step 2: Build execution plan
    plan = []
    for ctx in context_sizes:
        for pos in positions:
            for it in range(1, iterations + 1):
                sig = f"{ctx}_{pos}" if iterations == 1 else f"{ctx}_{pos}_run_{it}"
                plan.append({
                    "context_size": ctx,
                    "position": pos,
                    "iteration": it,
                    "run_label": f"run_{it}",
                    "signature": sig
                })

    total_runs = len(plan)
    samples_per_run = limit if limit is not None else 100
    total_samples = total_runs * samples_per_run
    total_api_calls = total_samples * 2

    print("=" * 78)
    print(" EXPERIMENT A2: MASTER EVALUATION MATRIX")
    print("=" * 78)
    print(f" Context Sizes  : {', '.join(context_sizes)} (Count: {len(context_sizes)})")
    print(f" Positions      : {', '.join(positions)} (Count: {len(positions)})")
    print(f" Iterations     : {iterations} per condition")
    print(f" Total Runs     : {total_runs} runs")
    print(f" Samples/Run    : {samples_per_run} prompts")
    print(f" Total Samples  : {total_samples:,} prompts")
    print(f" Total API Calls: {total_api_calls:,} requests (1 target + 1 judge call)")
    print(f" Runner Mode    : {runner.upper()} (Child Process Isolation with Inspect View logs)")
    print(f" Target Model   : {TARGET_MODEL}")
    print(f" Safety Judge   : {JUDGE_MODEL}")
    print(f" Fail Logger    : {FAIL_LOGGER_PATH.name}")
    print(f" Master CSV     : {MASTER_CSV_PATH.name}")
    print(f" Pacing Delay   : {pacing_delay:.1f}s between runs")
    print("=" * 78)

    # Step 3: Check completed runs in master CSV
    completed_signatures = get_completed_runs(MASTER_CSV_PATH, min_samples=samples_per_run)
    print(f"[Resume Check] Detected {len(completed_signatures)} completed runs in {MASTER_CSV_PATH.name}")

    runs_to_execute = []
    for item in plan:
        sig = item["signature"]
        if sig in completed_signatures and not force_rerun:
            print(f"   [SKIP - ALREADY DONE] {item['context_size']:<5} | {item['position']:<6} | {item['run_label']}")
        else:
            runs_to_execute.append(item)

    if max_runs is not None and max_runs > 0:
        total_pending = len(runs_to_execute)
        runs_to_execute = runs_to_execute[:max_runs]
        print(f"[Run Limiter] Limiting execution to first {len(runs_to_execute)} pending run(s) (--max-runs {max_runs}, of {total_pending} available)")

    print(f"\n[Plan Summary] Runs to execute: {len(runs_to_execute)} of {total_runs} (Skipped: {total_runs - len(runs_to_execute)})")
    needed_requests = len(runs_to_execute) * samples_per_run
    print(f"[Quota Required] Target calls: {needed_requests:,} | Judge calls: {needed_requests:,}")

    if dry_run:
        print("\n[DRY RUN COMPLETE] No API calls were made. Pass without --dry-run to start execution.\n")
        return

    if not runs_to_execute:
        print("\nAll scheduled runs are already completed in master_eval_results.csv!")
        print("Use --force to re-run completed runs if desired.\n")
        return

    # Load master prompt template for direct runner
    master_prompt_template = ""
    if runner == "direct":
        with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
            master_prompt_template = f.read()

    # Step 4: Execute runs sequentially
    completed_in_session = 0
    failures_in_session = 0

    try:
        for idx, run_item in enumerate(runs_to_execute, start=1):
            ctx = run_item["context_size"]
            pos = run_item["position"]
            it = run_item["iteration"]

            if runner == "inspect":
                run_result = execute_single_run_inspect_process(
                    run_idx=idx,
                    total_runs=len(runs_to_execute),
                    context_size=ctx,
                    position=pos,
                    iteration=it,
                    limit=limit,
                    fail_logger_path=FAIL_LOGGER_PATH,
                    master_csv_path=MASTER_CSV_PATH,
                    force_rerun=force_rerun
                )
            else:
                run_result = execute_single_run(
                    run_idx=idx,
                    total_runs=len(runs_to_execute),
                    context_size=ctx,
                    position=pos,
                    iteration=it,
                    master_prompt_template=master_prompt_template,
                    limit=limit,
                    pacing_delay=pacing_delay,
                    fail_logger_path=FAIL_LOGGER_PATH,
                    master_csv_path=MASTER_CSV_PATH
                )

            completed_in_session += 1
            failures_in_session += run_result.get("failed_samples", 0)

            # Telemetry update
            pct = (completed_in_session / len(runs_to_execute)) * 100
            print(f">>> Progress: {completed_in_session}/{len(runs_to_execute)} runs executed ({pct:.1f}%) | "
                  f"Failures logged: {failures_in_session}")

            # Pacing delay between runs
            if idx < len(runs_to_execute):
                print(f"[Run Cooldown] Sleeping {pacing_delay}s before starting next run...\n")
                time.sleep(pacing_delay)

    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print("[KEYBOARD INTERRUPT] Master evaluation suite stopped by user.")
        print(f"All progress up to the last evaluated sample is saved in {MASTER_CSV_PATH.name}.")
        print("Re-running master-eval.py will seamlessly resume from where you left off.")
        print("=" * 78 + "\n")
        sys.exit(130)

    total_duration = time.time() - start_suite_time
    print("\n" + "=" * 78)
    print(" MASTER EVALUATION SUITE COMPLETE")
    print("=" * 78)
    print(f" Executed Runs      : {completed_in_session} of {len(runs_to_execute)}")
    print(f" Failed Samples     : {failures_in_session} (Logged to {FAIL_LOGGER_PATH.name})")
    print(f" Total Time Taken   : {total_duration / 60:.2f} minutes")
    print(f" Master CSV Updated : {MASTER_CSV_PATH.resolve()}")
    print("=" * 78 + "\n")


# ==============================================================================
# 6. CLI INTERFACE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Master orchestration pipeline for Experiment A2 (15 runs: 5 context sizes x 3 positions x 1 run)"
    )
    parser.add_argument(
        "--context-sizes",
        nargs="+",
        default=BENCHMARK_CONTEXT_SIZES,
        help="Context sizes to evaluate (default: 1k 5k 10k 25k 50k)"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=BENCHMARK_POSITIONS,
        help="Positions to evaluate (default: start end middle)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of iterations per position and context size (default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit sample count per run for rapid testing (e.g. --limit 5, default: 100)"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Execute only the first N pending runs from the suite (e.g. --max-runs 5), then stop cleanly."
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between sample requests to stay safely under RPM limits (default: 2.0s)"
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="inspect",
        choices=["inspect", "direct"],
        help="Execution runner: 'inspect' spawns dedicated child process with real-time Inspect AI logs viewable in `inspect view` (default: inspect); 'direct' runs in-process loop."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if run signature already exists in master_eval_results.csv"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview full 45-run schedule, check quota status, and exit without sending API requests"
    )
    parser.add_argument(
        "--refresh-keys",
        action="store_true",
        help="Verify and display active API key configuration status only"
    )

    args = parser.parse_args()

    # If only refreshing keys was requested
    if args.refresh_keys:
        refresh_and_display_api_keys()
        return

    # Normalize context sizes & positions
    norm_contexts = [str(c).lower().strip() for c in args.context_sizes]
    norm_positions = [str(p).lower().strip() for p in args.positions]

    run_master_evaluation_suite(
        context_sizes=norm_contexts,
        positions=norm_positions,
        iterations=args.iterations,
        limit=args.limit,
        max_runs=args.max_runs,
        runner=args.runner,
        pacing_delay=args.pacing_delay,
        force_rerun=args.force,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
