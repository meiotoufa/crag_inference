"""
reward.py - Evaluation/reward computation using GPT-4 judge.
Adapted from local_evaluation.py in the CRAG benchmark.

Scoring:
  correct   → +1
  missing   →  0  ("I don't know")
  incorrect → -1  (hallucination)

Overall score = (2 * n_correct + n_miss) / n - 1
"""
import asyncio
import json
import re
import sys
import os
from typing import Dict, List, Tuple

from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from loguru import logger

# Import evaluation prompts from the original CRAG repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS


def get_system_message() -> str:
    """Returns the evaluation system message with instructions and examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def parse_response(response: str) -> Tuple[str, int]:
    """
    Parse GPT-4 judge response.
    Returns (explanation, score) where score is 0 (wrong) or 1 (correct).
    Returns score=-1 on parse failure.
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"

    try:
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score not in (0, 1):
                return "Bad score", -1
        else:
            return "Score not found", -1

        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        explanation = explanation_match.group(1) if explanation_match else text
        return explanation, score
    except Exception as e:
        return str(e), -1


async def evaluate_single(
    client: AsyncOpenAI,
    model_name: str,
    query: str,
    ground_truth: str,
    prediction: str,
    system_message: str,
    max_retries: int = 3,
) -> int:
    """
    Evaluate a single prediction against ground truth.
    Returns:
      1  = correct
      0  = hallucination/incorrect
     -1  = parse error (treated as incorrect)
     -2  = missing ("i don't know")
    """
    prediction_lower = prediction.lower().strip()
    ground_truth_lower = ground_truth.lower().strip()

    # "I don't know" → missing
    if "i don't know" in prediction_lower:
        return -2

    # Exact match
    if prediction_lower == ground_truth_lower:
        return 1

    # Both say invalid
    if "invalid" in prediction_lower and "invalid" in ground_truth_lower:
        return 1

    # One says invalid, other doesn't
    if "invalid" in prediction_lower or "invalid" in ground_truth_lower:
        return 0

    # Use GPT-4 judge
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
        },
    ]

    for _ in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            _, score = parse_response(content)
            return score
        except (APIConnectionError, RateLimitError):
            continue
        except Exception as e:
            logger.error(f"Eval API error: {e}")
            break

    return -1


async def evaluate_batch(
    client: AsyncOpenAI,
    model_name: str,
    queries: List[str],
    ground_truths: List[str],
    predictions: List[str],
    system_message: str,
) -> List[int]:
    """Evaluate a batch of predictions concurrently."""
    tasks = [
        evaluate_single(client, model_name, q, gt, pred, system_message)
        for q, gt, pred in zip(queries, ground_truths, predictions)
    ]
    return await asyncio.gather(*tasks)


async def evaluate_topic_results_async(
    topic: str,
    queries: List[str],
    ground_truths: List[str],
    predictions: List[str],
    config: dict,
) -> Dict:
    """
    Evaluate all predictions for one topic using async batch parallelism.

    Returns dict with score, accuracy, hallucination_rate, missing_rate, counts.
    """
    eval_config = config.get("eval_model", {})
    model_name = eval_config.get("name", "gpt-4-0125-preview")

    client_kwargs = {}
    if eval_config.get("base_url"):
        client_kwargs["base_url"] = eval_config["base_url"]
    if eval_config.get("api_key"):
        client_kwargs["api_key"] = eval_config["api_key"]

    client = AsyncOpenAI(**client_kwargs)
    system_message = get_system_message()

    scores = await evaluate_batch(
        client, model_name, queries, ground_truths, predictions, system_message
    )

    n_correct = sum(1 for s in scores if s == 1)
    n_miss = sum(1 for s in scores if s == -2)
    n_hallucination = sum(1 for s in scores if s not in (1, -2))

    n = len(predictions)
    overall_score = (2 * n_correct + n_miss) / n - 1 if n > 0 else 0

    return {
        "topic": topic,
        "score": overall_score,
        "accuracy": n_correct / n if n > 0 else 0,
        "hallucination_rate": n_hallucination / n if n > 0 else 0,
        "missing_rate": n_miss / n if n > 0 else 0,
        "n_correct": n_correct,
        "n_miss": n_miss,
        "n_hallucination": n_hallucination,
        "total": n,
    }


def evaluate_topic_results(
    topic: str,
    queries: List[str],
    ground_truths: List[str],
    predictions: List[str],
    config: dict,
) -> Dict:
    """Sync wrapper for evaluate_topic_results_async."""
    return asyncio.run(
        evaluate_topic_results_async(topic, queries, ground_truths, predictions, config)
    )
