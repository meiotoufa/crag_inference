#!/bin/bash
# distill.sh - Extract successful tool-calling traces for training data
#
# This script post-processes evaluation results to create training examples
# from successful multi-turn interactions (score >= 1).
#
# Usage:
#   bash example/distill.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Distillation script - TODO: implement trace extraction"
echo "This will read results/test_results/*.json and extract"
echo "successful conversations into data/train_test/"
