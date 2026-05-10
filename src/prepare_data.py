"""
prepare_data.py - Convert CRAG Task 3 data to per-topic JSONL for the evaluation pipeline.

Steps:
1. Combine tar.bz2 parts into a single archive
2. Extract the JSONL file(s) from the tar
3. Parse each line and group by domain
4. Write per-topic JSONL files to data/processed/

Usage:
    cd my_code/
    python -m src.prepare_data --version v5

    # Or specify custom paths:
    python -m src.prepare_data \
        --data_dir ../data \
        --output_dir data/processed \
        --version v5
"""
import argparse
import bz2
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path


def combine_parts(data_dir: str, version: str) -> str:
    """Combine split tar.bz2 parts into a single file. Returns path to combined file."""
    pattern = f"crag_task_3_dev_{version}.tar.bz2.part"
    parts = sorted(
        [f for f in os.listdir(data_dir) if f.startswith(pattern.rstrip("part")) and "part" in f]
    )

    if not parts:
        raise FileNotFoundError(f"No Task 3 {version} parts found in {data_dir}")

    combined_path = os.path.join(data_dir, f"crag_task_3_dev_{version}.tar.bz2")

    # Skip if already combined
    if os.path.exists(combined_path):
        print(f"Combined file already exists: {combined_path}")
        return combined_path

    print(f"Combining {len(parts)} parts: {parts}")
    with open(combined_path, "wb") as out:
        for part in parts:
            part_path = os.path.join(data_dir, part)
            with open(part_path, "rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"  merged {part}")

    print(f"Combined file: {combined_path}")
    return combined_path


def extract_and_group(tar_path: str) -> dict:
    """
    Extract JSONL from the tar.bz2 archive, group items by domain.
    Returns {domain: [item, ...]}
    """
    groups = defaultdict(list)
    total = 0

    print(f"Opening archive: {tar_path}")
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Task 3 contains JSONL file(s)
            if member.name.endswith(".jsonl") or member.name.endswith(".json"):
                print(f"  extracting: {member.name} ({member.size / 1024 / 1024:.1f} MB)")
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    domain = item.get("domain", "open")
                    groups[domain].append(item)
                    total += 1
                    if total % 100 == 0:
                        print(f"    processed {total} items...", end="\r")

    print(f"\nTotal items extracted: {total}")
    for domain, items in sorted(groups.items()):
        print(f"  {domain}: {len(items)} questions")

    return groups


def extract_bz2_jsonl(bz2_path: str) -> dict:
    """Extract from a plain .jsonl.bz2 file (Task 1&2 format)."""
    groups = defaultdict(list)
    total = 0

    print(f"Opening: {bz2_path}")
    with bz2.open(bz2_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            domain = item.get("domain", "open")
            groups[domain].append(item)
            total += 1
            if total % 100 == 0:
                print(f"  processed {total} items...", end="\r")

    print(f"\nTotal items: {total}")
    for domain, items in sorted(groups.items()):
        print(f"  {domain}: {len(items)} questions")

    return groups


def write_topic_files(groups: dict, output_dir: str, strip_html: bool = False):
    """Write per-topic JSONL files to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for domain, items in sorted(groups.items()):
        output_path = os.path.join(output_dir, f"{domain}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                # Optionally strip full HTML to reduce file size
                if strip_html and "search_results" in item:
                    for sr in item["search_results"]:
                        sr.pop("page_result", None)

                # Normalize field names: alt_ans → alternative_answers
                if "alt_ans" in item and "alternative_answers" not in item:
                    item["alternative_answers"] = item.pop("alt_ans")

                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Written: {output_path} ({len(items)} items, {size_mb:.1f} MB)")

    # Also write a combined file
    all_path = os.path.join(output_dir, "all_topics.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for domain in sorted(groups.keys()):
            for item in groups[domain]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    total = sum(len(v) for v in groups.values())
    size_mb = os.path.getsize(all_path) / 1024 / 1024
    print(f"Written: {all_path} ({total} items, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert CRAG Task 3 data to per-topic JSONL")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        help="Directory containing raw CRAG data files",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Output directory for per-topic JSONL files",
    )
    parser.add_argument(
        "--version",
        default="v5",
        choices=["v4", "v5"],
        help="Dataset version (v4 or v5)",
    )
    parser.add_argument(
        "--task",
        default="3",
        choices=["1_and_2", "3"],
        help="Which task data to convert (default: 3)",
    )
    parser.add_argument(
        "--strip_html",
        action="store_true",
        help="Remove full HTML (page_result) from search_results to reduce file size",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Version: {args.version}, Task: {args.task}")
    print(f"Strip HTML: {args.strip_html}")
    print("=" * 50)

    if args.task == "3":
        # Task 3: tar.bz2 parts → combine → extract
        combined_path = combine_parts(data_dir, args.version)
        groups = extract_and_group(combined_path)
    else:
        # Task 1&2: plain .jsonl.bz2
        bz2_path = os.path.join(data_dir, f"crag_task_1_and_2_dev_{args.version}.jsonl.bz2")
        if not os.path.exists(bz2_path):
            raise FileNotFoundError(f"File not found: {bz2_path}")
        groups = extract_bz2_jsonl(bz2_path)

    write_topic_files(groups, output_dir, strip_html=args.strip_html)
    print("\nDone!")


if __name__ == "__main__":
    main()
