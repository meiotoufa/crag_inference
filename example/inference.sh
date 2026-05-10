#!/bin/bash
# inference.sh - Run CRAG evaluation framework
#
# Prerequisites:
#   1. vLLM model server running (default: http://localhost:8000/v1)
#   2. CRAG mock_api server running (default: http://localhost:8001)
#      Start it with: cd mock_api && uvicorn server:app --host 0.0.0.0 --port 8001
#   3. OPENAI_API_KEY set for GPT-4 evaluation
#
# Usage:
#   bash example/inference.sh [config_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG=${1:-"config.yaml"}

echo "============================================"
echo "CRAG Evaluation Framework"
echo "Config: $CONFIG"
echo "============================================"

# Read URLs from config
MOCK_API_URL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['mock_api']['base_url'])")
VLLM_URL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['base_url'])")

# Check mock_api
echo "Checking mock_api at $MOCK_API_URL ..."
if ! curl -s "$MOCK_API_URL/" > /dev/null 2>&1; then
    echo "ERROR: Mock API server not running at $MOCK_API_URL"
    echo "Start it with: cd ../mock_api && uvicorn server:app --host 0.0.0.0 --port 8001"
    exit 1
fi
echo "  -> mock_api OK"

# Check vLLM
echo "Checking vLLM at $VLLM_URL ..."
if ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at $VLLM_URL"
    exit 1
fi
echo "  -> vLLM OK"

echo ""
echo "Starting evaluation..."
python3 -m src.test --config "$CONFIG"

echo "============================================"
echo "Evaluation complete. Results in results/"
echo "============================================"
