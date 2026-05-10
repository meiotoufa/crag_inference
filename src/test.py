"""
test.py - Main evaluation entry point.

Usage:
    cd template/
    python -m src.test --config config.yaml
"""
import argparse
import asyncio
import bz2
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger

from src.call_llm import LLMClient, ConversationResult
from src.reward import evaluate_topic_results


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> List[dict]:
    """Load JSONL data (supports .jsonl and .jsonl.bz2)."""
    items = []
    if data_path.endswith(".bz2"):
        opener = bz2.open
    else:
        opener = open
    with opener(data_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def group_by_topic(items: List[dict]) -> Dict[str, List[dict]]:
    """Group data items by domain (topic)."""
    groups = defaultdict(list)
    for item in items:
        domain = item.get("domain", "open")
        groups[domain].append(item)
    return groups


def load_system_prompt(prompts_dir: str = "prompts") -> str:
    """Load system prompt template."""
    prompt_path = os.path.join(prompts_dir, "system_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()


async def run_topic(
    topic: str,
    questions: List[dict],
    llm_client: LLMClient,
    system_prompt: str,
    concurrency: int,
) -> List[ConversationResult]:
    """Run all questions for a single topic with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_question(item: dict) -> ConversationResult:
        async with semaphore:
            return await llm_client.run_react_loop(
                interaction_id=item["interaction_id"],
                query=item["query"],
                domain=item.get("domain", topic),
                query_time=item.get("query_time", ""),
                system_prompt=system_prompt,
            )

    tasks = [process_question(item) for item in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Error on {questions[i]['interaction_id']}: {r}")
            processed.append(ConversationResult(
                interaction_id=questions[i]["interaction_id"],
                query=questions[i]["query"],
                final_answer="I don't know",
                turns_used=0,
                error=str(r),
            ))
        else:
            processed.append(r)

    return processed


def save_results(topic: str, results: List[ConversationResult], output_dir: str):
    """Save per-topic results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{topic}_results.json")

    data = []
    for r in results:
        data.append({
            "interaction_id": r.interaction_id,
            "query": r.query,
            "prediction": r.final_answer,
            "turns_used": r.turns_used,
            "error": r.error,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(data)} results to {output_path}")


async def main(config_path: str):
    config = load_config(config_path)

    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, level=log_level)

    system_prompt = load_system_prompt()
    llm_client = LLMClient(config)

    # Load data
    data_path = config["data"]["input_path"]
    all_items = []
    if os.path.isfile(data_path):
        all_items = load_data(data_path)
    else:
        for f in sorted(Path(data_path).glob("*.jsonl*")):
            all_items.extend(load_data(str(f)))

    if not all_items:
        logger.error(f"No data found at {data_path}")
        return

    topic_groups = group_by_topic(all_items)
    topics_to_eval = config.get("topics", list(topic_groups.keys()))
    concurrency = config["inference"]["batch_concurrency"]
    results_dir = config["output"]["results_dir"]
    stats_dir = config["output"]["statistics_dir"]

    logger.info(f"Loaded {len(all_items)} questions across {len(topic_groups)} topics")
    logger.info(f"Topics to evaluate: {topics_to_eval}")

    all_stats = {}

    for topic in topics_to_eval:
        if topic not in topic_groups:
            logger.warning(f"Topic '{topic}' not found in data, skipping.")
            continue

        questions = topic_groups[topic]
        logger.info(f"[{topic}] Processing {len(questions)} questions (concurrency={concurrency})...")

        start_time = time.time()
        results = await run_topic(topic, questions, llm_client, system_prompt, concurrency)
        elapsed = time.time() - start_time
        logger.info(f"[{topic}] Inference done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} q/s)")

        # Save raw results
        save_results(topic, results, results_dir)

        # Run evaluation
        predictions = [r.final_answer for r in results]
        ground_truths = [q["answer"] for q in questions]
        queries = [q["query"] for q in questions]

        logger.info(f"[{topic}] Running GPT-4 evaluation...")
        stats = evaluate_topic_results(
            topic=topic,
            queries=queries,
            ground_truths=ground_truths,
            predictions=predictions,
            config=config,
        )
        stats["elapsed_seconds"] = elapsed
        stats["num_questions"] = len(questions)
        all_stats[topic] = stats

        logger.info(
            f"[{topic}] score={stats['score']:.3f} "
            f"acc={stats['accuracy']:.3f} "
            f"hall={stats['hallucination_rate']:.3f} "
            f"miss={stats['missing_rate']:.3f}"
        )

    # Save overall statistics
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "evaluation_summary.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Overall statistics saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAG Evaluation Framework")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config))
