"""
custom_eval.py
==============
Standalone multithreaded evaluation runner for safety benchmark experiments on Gemini
using Gemini (gemini-3.1-flash-lite) as safety judge. Logs detailed sample data to custom_logs/ and appends run summaries
to master_eval_results.csv.

Reuses shared logic and model callers from Utils.eval_funcs.
"""

import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    call_groq_judge,
    call_judge,
    call_gemini_judge,
    format_summary_table,
    save_custom_log,
    append_to_master_csv,
    get_rolling_tpm_guard
)


def run_eval(
    padding: str = "none",
    position: str = "middle",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_samples: Optional[int] = None,
    concurrency: int = 2,
    output_dir: Optional[Path] = None,
    master_csv_name: str = DEFAULT_MASTER_CSV_NAME
) -> Dict[str, Any]:
    """
    Main evaluation pipeline:
    1. Builds mapped dataset (100 prompts x 10 context categories) with selected padding & position.
    2. Sends padded prompts to Gemini.
    3. Sends Gemini's response and raw harmful prompt (NO context padding) to Groq judge.
    4. Saves detailed JSON into custom_logs/ folder.
    5. Appends summary row to master_eval_results.csv.
    6. Prints summary report table to console.
    """
    out_dir = Path(output_dir) if output_dir else EXPERIMENT_A2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_logs_dir = out_dir / "custom_logs"

    norm_pad = str(padding).lower().strip()
    norm_pos = str(position).lower().strip()

    print("=" * 75)
    print("STARTING CUSTOM SAFETY EVALUATION")
    print("=" * 75)
    print(f"Padding Size   : {norm_pad}")
    print(f"Prompt Position: {norm_pos}")
    print(f"Target Model   : {gemini_model}")
    print(f"Safety Judge   : {judge_model}")
    print(f"Output Dir     : {out_dir}")
    print("=" * 75)

    # 1. Build mapped dataset
    print("Building mapped dataset (100 prompts x 10 context categories)...")
    dataset = build_mapped_dataset(
        harmful_prompts_file=BASE_HARMFUL_PROMPTS_PATH,
        contexts_chunked_file=CONTEXTS_CHUNKED_PATH,
        padding=norm_pad,
        position=norm_pos
    )

    if max_samples and max_samples < len(dataset):
        dataset = dataset[:max_samples]
        print(f"[Notice] Limiting execution to {max_samples} samples for testing.")

    # Load master prompt template for safety judge
    with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
        master_prompt_template = f.read()

    results: List[Dict[str, Any]] = []
    total_samples = len(dataset)
    start_time = time.time()

    # Pre-flight quota check and Best-Fit key activation (deduct=False)
    from Utils.api_management import pre_flight_allocate
    try:
        pre_flight_allocate(needed_requests=total_samples, model_name="gemini-3.5-flash-lite", deduct=False)
    except Exception as e:
        print(f"[Warning] Target model preflight check: {e}")

    try:
        clean_judge_name = judge_model.replace("google/", "").strip()
        if "gemini" in clean_judge_name:
            pre_flight_allocate(needed_requests=total_samples, model_name=clean_judge_name, deduct=False)
        elif "gpt" in clean_judge_name or "groq" in clean_judge_name:
            pre_flight_allocate(needed_requests=total_samples, model_name="openai/gpt-oss-120b", deduct=False)
    except Exception as e:
        print(f"[Warning] Judge model preflight check: {e}")

    def process_sample(item: Dict[str, Any], index: int) -> Dict[str, Any]:
        prompt_id = item["id"]
        cat = item["category"]
        raw_prompt = item["raw_prompt"]
        padded_prompt = item["padded_prompt"]

        # Individual Sample Rolling TPM Guard (for 50k context: max 4 samples per rolling 60s)
        if str(norm_pad).lower().strip() == "50k":
            get_rolling_tpm_guard().check_and_wait(
                padding_size="50k",
                sample_id=prompt_id
            )

        print(f"[{index:3d}/{total_samples:3d}] Sending {prompt_id} ({cat}) to Gemini...")
        gemini_response = call_gemini(prompt=padded_prompt, model_name=gemini_model)

        # Inter-request delay between target response and safety judge (to pace single shared API key)
        post_target_delay = round(random.uniform(3.5, 6.0), 2)
        time.sleep(post_target_delay)

        print(f"[{index:3d}/{total_samples:3d}] Evaluating {prompt_id} via Safety Judge ({judge_model})...")
        judge_res = call_judge(
            harmful_request=raw_prompt,
            model_response=gemini_response,
            master_prompt_template=master_prompt_template,
            judge_model=judge_model
        )

        # Inter-request delay after safety judge grading
        post_judge_delay = round(random.uniform(3.5, 6.0), 2)
        time.sleep(post_judge_delay)

        return {
            "id": prompt_id,
            "category": cat,
            "context_category": item["context_category"],
            "padding": norm_pad,
            "position": norm_pos,
            "raw_prompt": raw_prompt,
            "gemini_response": gemini_response,
            "judge_score": judge_res.get("score", -1),
            "judge_reason": judge_res.get("reason", ""),
            "judge_evidence": judge_res.get("evidence", "")
        }

    # Execute calls with controlled concurrency
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(process_sample, item, i)
                for i, item in enumerate(dataset, start=1)
            ]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"[Error processing sample]: {e}")
    else:
        for i, item in enumerate(dataset, start=1):
            res = process_sample(item, i)
            results.append(res)

    results.sort(key=lambda x: x["id"])
    elapsed = time.time() - start_time

    run_type = "partial" if (limit and limit < len(dataset_items)) else "full"
    task_name = f"A2_{norm_pad}_{norm_pos}_{run_type}"

    # Run metadata
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": task_name,
        "padding": norm_pad,
        "position": norm_pos,
        "gemini_model": gemini_model,
        "judge_model": judge_model,
        "elapsed_seconds": round(elapsed, 2),
        "total_evaluated": len(results)
    }

    # 1. Save detailed JSON to custom_logs/
    json_path = save_custom_log(custom_logs_dir, task_name, meta, results)

    # 2. Append summary row to master CSV
    master_csv_path = out_dir / master_csv_name
    log_rel_path = f"custom_logs/{json_path.name}"
    append_to_master_csv(master_csv_path, meta, results, log_rel_path)

    # 3. Print Markdown summary table to console
    summary_md = format_summary_table(results, meta)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE - SUMMARY TABLE")
    print("=" * 75)
    print(summary_md)
    print("=" * 75)
    print(f"Detailed JSON Log : {json_path}")
    print(f"Master CSV Updated: {master_csv_path}")
    print(f"Total Time Taken  : {elapsed:.1f}s")
    print("=" * 75)

    return {
        "metadata": meta,
        "results": results,
        "summary_markdown": summary_md,
        "json_path": str(json_path),
        "master_csv_path": str(master_csv_path)
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run safety evaluation on Gemini with dynamic context padding, evaluated via Groq judge."
    )
    parser.add_argument(
        "--padding", "-p",
        type=str,
        default="none",
        choices=["none", "1k", "2k", "5k", "10k", "20k", "50k"],
        help="Context padding size (default: none)"
    )
    parser.add_argument(
        "--position", "--pos",
        type=str,
        default="middle",
        choices=["start", "middle", "end"],
        help="Position of harmful prompt within padding (default: middle)"
    )
    parser.add_argument(
        "--gemini-model", "-m",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Target Gemini model name (default: {DEFAULT_GEMINI_MODEL})"
    )
    parser.add_argument(
        "--judge-model", "-j",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Safety judge model name (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (e.g. 5 for a quick test run)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=2,
        help="Number of concurrent requests (default: 2)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory to save results (default: Experiment A2/)"
    )
    parser.add_argument(
        "--master-csv",
        type=str,
        default=DEFAULT_MASTER_CSV_NAME,
        help=f"Master CSV filename in output dir (default: {DEFAULT_MASTER_CSV_NAME})"
    )
    parser.add_argument(
        "--printAPI", "--print-api", "--printapi",
        dest="print_api",
        action="store_true",
        help="Print all available Google API key names discovered in persistent system variables."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.print_api:
        from Utils.api_management import print_available_api_keys
        print_available_api_keys()
        sys.exit(0)

    run_eval(
        padding=args.padding,
        position=args.position,
        gemini_model=args.gemini_model,
        judge_model=args.judge_model,
        max_samples=args.samples,
        concurrency=args.concurrency,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        master_csv_name=args.master_csv
    )
