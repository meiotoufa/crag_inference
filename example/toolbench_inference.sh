#!/bin/bash
# toolbench_inference.sh - Run ToolBench evaluation framework
#
# Prerequisites:
#   1. vLLM model server running (default: http://localhost:9000/v1)
#   2. ToolBench data downloaded (run: bash example/download_toolbench.sh)
#   3. pip install datasets openai loguru pyyaml
#
# Usage:
#   bash example/toolbench_inference.sh [config_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG=${1:-"toolbench_config.yaml"}

echo "============================================"
echo "ToolBench Evaluation Framework"
echo "Config: $CONFIG"
echo "============================================"

# Read URLs from config
VLLM_URL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['base_url'])")

# Check if data exists
DATA_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['input_path'])")
echo "Checking data at $DATA_DIR ..."
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "WARNING: No ToolBench data found at $DATA_DIR"
    echo "Downloading data first..."
    echo ""
    bash example/download_toolbench.sh
    echo ""
fi
echo "  -> Data OK ($(ls "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l | tr -d ' ') split files found)"

# Check vLLM
echo "Checking vLLM at $VLLM_URL ..."
if ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at $VLLM_URL"
    echo "Start it with: python -m vllm.entrypoints.openai.api_server --model <model_name> --port 9000"
    exit 1
fi
echo "  -> vLLM OK"

# Show config summary
echo ""
echo "Configuration:"
echo "  Model:       $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['name'])")"
echo "  Max turns:   $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['inference']['max_turns'])")"
echo "  Concurrency: $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['inference']['batch_concurrency'])")"
echo "  Splits:      $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(', '.join(c.get('splits', ['g1_instruction'])))")"
echo ""

echo "Starting ToolBench evaluation..."
python3 -m src.toolbench_test --config "$CONFIG"

echo ""
echo "============================================"
echo "Evaluation complete."
echo "  Results:    $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output']['results_dir'])")"
echo "  Statistics: $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output']['statistics_dir'])")"
echo "============================================"
