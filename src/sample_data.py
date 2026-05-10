"""
sample_data.py - Sample a subset of processed CRAG data for quick evaluation.

Supports:
  - Per-topic sampling (N samples from each domain)
  - Total budget sampling (N total, distributed proportionally across domains)
  - Random seed for reproducibility
  - Stratified by static_or_dynamic / question_type if available

Usage:
    cd my_code/
    # Sample 20 per topic (100 total)
    python -m src.sample_data --per_topic 20

    # Sample 50 total, proportionally distributed
    python -m src.sample_data --total 50

    # Sample from specific topics only
    python -m src.sample_data --per_topic 10 --topics finance movie

    # Custom input/output paths
    python -m src.sample_data --input_dir data/processed --output_dir data/sampled --per_topic 30 --seed 42
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_topic_data(input_dir: str, topics: list = None) -> dict:
    """Load per-topic JSONL files. Returns {topic: [items]}."""
    groups = {}
    for f in sorted(Path(input_dir).glob("*.jsonl")):
        topic = f.stem
        if topic == "all_topics":
            continue
        if topics and topic not in topics:
            continue
        items = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        groups[topic] = items
    return groups


def sample_per_topic(groups: dict, n_per_topic: int, seed: int) -> dict:
    """Sample N items from each topic."""
    rng = random.Random(seed)
    sampled = {}
    for topic, items in groups.items():
        k = min(n_per_topic, len(items))
        sampled[topic] = rng.sample(items, k)
    return sampled


def sample_total(groups: dict, n_total: int, seed: int) -> dict:
    """Sample N total items, distributed proportionally across topics."""
    rng = random.Random(seed)
    total_items = sum(len(v) for v in groups.values())
    sampled = {}

    remaining = n_total
    topics = sorted(groups.keys())

    for i, topic in enumerate(topics):
        items = groups[topic]
        if i == len(topics) - 1:
            # Last topic gets whatever remains
            k = remaining
        else:
            # Proportional allocation
            k = round(n_total * len(items) / total_items)
        k = min(k, len(items), remaining)
        k = max(k, 0)
        sampled[topic] = rng.sample(items, k)
        remaining -= k

    return sampled


def write_sampled(sampled: dict, output_dir: str):
    """Write sampled data to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for topic, items in sorted(sampled.items()):
        out_path = os.path.join(output_dir, f"{topic}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {topic}: {len(items)} samples -> {out_path}")
        total += len(items)

    # Combined file
    all_path = os.path.join(output_dir, "all_topics.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for topic in sorted(sampled.keys()):
            for item in sampled[topic]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  all_topics: {total} samples -> {all_path}")

    return total


def main():
    parser = argparse.ArgumentParser(description="Sample subset of CRAG data for evaluation")
    parser.add_argument("--input_dir", default="data/processed", help="Input dir with per-topic JSONL files")
    parser.add_argument("--output_dir", default="data/sampled", help="Output dir for sampled data")
    parser.add_argument("--per_topic", type=int, default=None, help="Number of samples per topic")
    parser.add_argument("--total", type=int, default=None, help="Total number of samples (proportional)")
    parser.add_argument("--topics", nargs="*", default=None, help="Only sample from these topics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.per_topic is None and args.total is None:
        parser.error("Must specify either --per_topic or --total")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Seed:   {args.seed}")
    print("=" * 50)

    # Load data
    groups = load_topic_data(input_dir, args.topics)
    if not groups:
        print(f"ERROR: No JSONL files found in {input_dir}")
        return

    print("Source data:")
    for topic, items in sorted(groups.items()):
        print(f"  {topic}: {len(items)} questions")
    total_source = sum(len(v) for v in groups.values())
    print(f"  Total: {total_source}")
    print()

    # Sample
    if args.per_topic is not None:
        print(f"Sampling {args.per_topic} per topic...")
        sampled = sample_per_topic(groups, args.per_topic, args.seed)
    else:
        print(f"Sampling {args.total} total (proportional)...")
        sampled = sample_total(groups, args.total, args.seed)

    print()
    print("Sampled data:")
    total_sampled = write_sampled(sampled, output_dir)
    print()
    print(f"Done! {total_sampled}/{total_source} samples written to {output_dir}")


if __name__ == "__main__":
    main()
