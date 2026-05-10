# CRAG Evaluation Framework

Multi-turn tool-calling evaluation for the CRAG benchmark. Uses ReAct-style agent loop with vLLM-served models.

## Quick Start

```bash
# 1. Start CRAG mock_api (knowledge graph server)
cd ../mock_api && uvicorn server:app --host 0.0.0.0 --port 8001

# 2. Start vLLM model server
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000

# 3. Set OpenAI API key (for GPT-4 evaluation)
export OPENAI_API_KEY=sk-...

# 4. Prepare data
# Place JSONL files in data/processed/ (one or multiple files)

# 5. Run evaluation
bash example/inference.sh
```

## Config

Edit `config.yaml` to change:
- `model.name` / `model.base_url` — vLLM model endpoint
- `inference.max_turns` — max ReAct loop iterations per question
- `inference.batch_concurrency` — parallel questions per topic
- `mock_api.base_url` — CRAG knowledge graph API address
- `topics` — which domains to evaluate

## Architecture

```
Question → [System Prompt + Tools] → Model → tool_calls? 
    → Yes: execute tool → feed result → loop
    → No (answer tool called): extract answer → evaluate
```

Each question runs through a ReAct loop where the model autonomously decides which KG tools to call. The loop terminates when the model calls the `answer` tool.

## Results

- `results/test_results/{topic}_results.json` — per-question predictions
- `results/statistic/evaluation_summary.json` — per-topic scores

## Scoring

- **Correct** (+1): prediction matches ground truth
- **Missing** (0): model says "I don't know"
- **Hallucination** (-1): wrong answer

Overall: `score = (2 * correct + missing) / total - 1`
