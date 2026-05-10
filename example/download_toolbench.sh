#!/bin/bash
# download_toolbench.sh - Download ToolBench evaluation data from HuggingFace
#
# Prerequisites:
#   pip install datasets huggingface_hub
#
# Usage:
#   bash example/download_toolbench.sh [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUTPUT_DIR=${1:-"data/toolbench"}

echo "============================================"
echo "ToolBench Data Download"
echo "Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR/raw"
mkdir -p "$OUTPUT_DIR/processed"

# Download ToolBench benchmark data from HuggingFace
python3 - <<'PYEOF'
import os
import json
import sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/toolbench"
raw_dir = os.path.join(output_dir, "raw")
processed_dir = os.path.join(output_dir, "processed")

print("Downloading ToolBench data from HuggingFace...")
from datasets import load_dataset

# Download benchmark splits (G1, G2, G3 evaluation sets)
benchmark_splits = [
    "g1_instruction",
    "g1_category",
    "g1_tool",
    "g2_instruction",
    "g2_category",
    "g3_instruction",
]

for split_name in benchmark_splits:
    print(f"  Downloading split: {split_name} ...")
    try:
        ds = load_dataset("tuandunghcmut/toolbench-v1", "benchmark", split=split_name)
        out_path = os.path.join(raw_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    -> {len(ds)} samples saved to {out_path}")
    except Exception as e:
        print(f"    [WARN] Failed to download {split_name}: {e}")

# Also download a subset of training data for reference
print("  Downloading training subset (first 1000 samples)...")
try:
    ds_train = load_dataset("tuandunghcmut/toolbench-v1", "default", split="train[:1000]")
    out_path = os.path.join(raw_dir, "train_subset.jsonl")
    with open(out_path, "w") as f:
        for item in ds_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"    -> {len(ds_train)} samples saved to {out_path}")
except Exception as e:
    print(f"    [WARN] Failed to download train subset: {e}")

# Process benchmark data into unified format for inference
print("\nProcessing benchmark data...")
total = 0
for split_name in benchmark_splits:
    raw_path = os.path.join(raw_dir, f"{split_name}.jsonl")
    if not os.path.exists(raw_path):
        continue
    out_path = os.path.join(processed_dir, f"{split_name}.jsonl")
    count = 0
    with open(raw_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            item = json.loads(line)
            # Normalize to unified format
            processed = {
                "query_id": item.get("query_id", f"{split_name}_{count}"),
                "query": item.get("query", ""),
                "api_list": json.loads(item["api_list"]) if isinstance(item.get("api_list"), str) else item.get("api_list", []),
                "relevant_apis": json.loads(item["relevant_apis"]) if isinstance(item.get("relevant_apis"), str) else item.get("relevant_apis", []),
                "split": split_name,
            }
            fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
            count += 1
    total += count
    print(f"  {split_name}: {count} samples processed")

print(f"\nDone! Total processed: {total} samples")
print(f"Raw data: {raw_dir}/")
print(f"Processed data: {processed_dir}/")
PYEOF "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Download complete!"
echo "  Raw:       $OUTPUT_DIR/raw/"
echo "  Processed: $OUTPUT_DIR/processed/"
echo "============================================"
