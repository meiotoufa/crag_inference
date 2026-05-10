"""
toolbench_test.py - Main entry point for ToolBench evaluation.

Usage:
    cd my_code/
    python -m src.toolbench_test --config toolbench_config.yaml
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger

from src.toolbench_call_llm import ToolBenchLLMClient, ToolBenchResult


def load_config(config_path: str = "toolbench_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_split_data(data_path: str) -> List[dict]:
    """Load JSONL data for a single split."""
    items = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_split(
    split_name: str,
    queries: List[dict],
    llm_client: ToolBenchLLMClient,
    concurrency: int,
) -> List[ToolBenchResult]:
    """Run all queries for a single split with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_query(item: dict) -> ToolBenchResult:
        async with semaphore:
            return await llm_client.run_react_loop(
                query_id=item["query_id"],
                query=item["query"],
                api_list=item.get("api_list", []),
            )

    tasks = [process_query(item) for item in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Error on {queries[i]['query_id']}: {r}")
            processed.append(ToolBenchResult(
                query_id=queries[i]["query_id"],
                query=queries[i]["query"],
                final_answer="",
                turns_used=0,
                status="error",
                error=str(r),
            ))
        else:
            processed.append(r)

    return processed


def save_results(split_name: str, results: List[ToolBenchResult], output_dir: str):
    """Save per-split results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}_results.json")

    data = []
    for r in results:
        data.append({
            "query_id": r.query_id,
            "query": r.query,
            "final_answer": r.final_answer,
            "turns_used": r.turns_used,
            "status": r.status,
            "error": r.error,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(data)} results to {output_path}")


def compute_pass_rate(results: List[ToolBenchResult]) -> Dict:
    """
    Compute ToolBench pass rate metrics.

    Pass: status == "success" and final_answer is non-empty
    Fail: status in ("error", "timeout", "max_turns", "give_up") or empty answer
    """
    n = len(results)
    n_pass = sum(1 for r in results if r.status == "success" and r.final_answer.strip())
    n_give_up = sum(1 for r in results if r.status == "give_up")
    n_error = sum(1 for r in results if r.status == "error")
    n_timeout = sum(1 for r in results if r.status == "timeout")
    n_max_turns = sum(1 for r in results if r.status == "max_turns")

    return {
        "total": n,
        "pass": n_pass,
        "pass_rate": n_pass / n if n > 0 else 0,
        "give_up": n_give_up,
        "error": n_error,
        "timeout": n_timeout,
        "max_turns": n_max_turns,
        "avg_turns": sum(r.turns_used for r in results) / n if n > 0 else 0,
    }


async def main(config_path: str):
    config = load_config(config_path)

    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, level=log_level)

    llm_client = ToolBenchLLMClient(config)

    data_dir = config["data"]["input_path"]
    splits_to_eval = config.get("splits", ["g1_instruction"])
    concurrency = config["inference"]["batch_concurrency"]
    results_dir = config["output"]["results_dir"]
    stats_dir = config["output"]["statistics_dir"]

    all_stats = {}

    for split_name in splits_to_eval:
        split_path = os.path.join(data_dir, f"{split_name}.jsonl")
        if not os.path.exists(split_path):
            logger.warning(f"Split file not found: {split_path}, skipping.")
            continue

        queries = load_split_data(split_path)
        if not queries:
            logger.warning(f"No data in {split_path}, skipping.")
            continue

        logger.info(f"[{split_name}] Processing {len(queries)} queries (concurrency={concurrency})...")

        start_time = time.time()
        results = await run_split(split_name, queries, llm_client, concurrency)
        elapsed = time.time() - start_time
        logger.info(f"[{split_name}] Inference done in {elapsed:.1f}s ({len(queries)/elapsed:.1f} q/s)")

        # Save raw results
        save_results(split_name, results, results_dir)

        # Compute pass rate
        stats = compute_pass_rate(results)
        stats["elapsed_seconds"] = elapsed
        stats["split"] = split_name
        all_stats[split_name] = stats

        logger.info(
            f"[{split_name}] pass_rate={stats['pass_rate']:.3f} "
            f"pass={stats['pass']}/{stats['total']} "
            f"give_up={stats['give_up']} "
            f"error={stats['error']} "
            f"avg_turns={stats['avg_turns']:.1f}"
        )

    # Save overall statistics
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "toolbench_summary.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Overall statistics saved to {stats_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Split':<20} {'Pass Rate':<12} {'Pass/Total':<12} {'Avg Turns':<10}")
    print("-" * 60)
    for split_name, stats in all_stats.items():
        print(f"{split_name:<20} {stats['pass_rate']:.3f}        {stats['pass']}/{stats['total']:<8} {stats['avg_turns']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToolBench Evaluation Framework")
    parser.add_argument("--config", default="toolbench_config.yaml", help="Path to config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
