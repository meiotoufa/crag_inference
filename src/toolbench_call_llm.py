"""
toolbench_call_llm.py - Async OpenAI client with multi-turn ReAct tool-calling loop for ToolBench.

Key differences from CRAG call_llm.py:
  - Tools are dynamic per query (from api_list), not fixed per domain.
  - "Finish" tool replaces "answer" tool with give_answer/give_up_and_restart semantics.
  - Tool executor handles simulated or real RapidAPI calls.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from src.toolbench_tools import ToolBenchExecutor, get_tools_for_query


@dataclass
class ToolBenchResult:
    """Result of a single query's ReAct loop."""
    query_id: str
    query: str
    final_answer: str
    turns_used: int
    messages: List[dict] = field(default_factory=list)
    status: str = "success"  # success, timeout, error, max_turns, give_up
    error: Optional[str] = None


TOOLBENCH_SYSTEM_PROMPT = """You are an AutoGPT-style AI assistant that can use tools to answer questions.

You have access to the following tools. Use them as needed to answer the user's question.

When you have enough information to answer, call the "Finish" tool with return_type="give_answer" and provide your final_answer.
If you cannot make progress, call "Finish" with return_type="give_up_and_restart".

Think step by step:
1. Analyze what information you need.
2. Call appropriate tools to gather information.
3. Synthesize the results.
4. Provide a final answer via the Finish tool.
"""


class ToolBenchLLMClient:
    """Async OpenAI client for ToolBench multi-turn inference."""

    def __init__(self, config: dict):
        self.client = AsyncOpenAI(
            base_url=config["model"]["base_url"],
            api_key=config["model"].get("api_key", "EMPTY"),
        )
        self.model_name = config["model"]["name"]
        self.temperature = config["model"].get("temperature", 0.1)
        self.top_p = config["model"].get("top_p", 0.9)
        self.max_tokens = config["model"].get("max_tokens", 4096)
        self.max_turns = config["inference"]["max_turns"]
        self.timeout = config["inference"].get("timeout_per_turn", 120)
        self.tool_executor = ToolBenchExecutor(config)

    async def run_react_loop(
        self,
        query_id: str,
        query: str,
        api_list: List[dict],
    ) -> ToolBenchResult:
        """
        Run the full ReAct loop for a ToolBench query.
        Terminates when:
          1. Model calls "Finish" with give_answer
          2. Model calls "Finish" with give_up_and_restart
          3. Model generates text without tool_calls (finish_reason=stop)
          4. max_turns exceeded
          5. timeout
        """
        # Register APIs for this query and build tool schemas
        self.tool_executor.register_apis(api_list)
        tools = get_tools_for_query(api_list)

        messages = [
            {"role": "system", "content": TOOLBENCH_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for turn in range(self.max_turns):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                return ToolBenchResult(
                    query_id=query_id,
                    query=query,
                    final_answer="",
                    turns_used=turn + 1,
                    messages=messages,
                    status="timeout",
                    error="timeout",
                )
            except Exception as e:
                return ToolBenchResult(
                    query_id=query_id,
                    query=query,
                    final_answer="",
                    turns_used=turn + 1,
                    messages=messages,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                )

            choice = response.choices[0]
            assistant_msg = choice.message
            messages.append(assistant_msg.model_dump())

            # No tool calls — model answered directly
            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                return ToolBenchResult(
                    query_id=query_id,
                    query=query,
                    final_answer=content.strip(),
                    turns_used=turn + 1,
                    messages=messages,
                    status="success",
                )

            # Process tool calls
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Check for Finish tool
                if fn_name == "Finish":
                    return_type = fn_args.get("return_type", "give_answer")
                    if return_type == "give_answer":
                        final_ans = fn_args.get("final_answer", "")
                        return ToolBenchResult(
                            query_id=query_id,
                            query=query,
                            final_answer=final_ans.strip(),
                            turns_used=turn + 1,
                            messages=messages,
                            status="success",
                        )
                    else:
                        return ToolBenchResult(
                            query_id=query_id,
                            query=query,
                            final_answer="",
                            turns_used=turn + 1,
                            messages=messages,
                            status="give_up",
                        )

                # Execute tool
                result_str = await asyncio.to_thread(
                    self.tool_executor.execute, fn_name, fn_args
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # Exhausted max turns
        return ToolBenchResult(
            query_id=query_id,
            query=query,
            final_answer="",
            turns_used=self.max_turns,
            messages=messages,
            status="max_turns",
            error="max_turns_exceeded",
        )
