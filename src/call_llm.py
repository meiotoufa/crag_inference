"""
call_llm.py - Async OpenAI client with multi-turn ReAct tool-calling loop.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from src.tools import ToolExecutor, get_tools_for_domain


@dataclass
class ConversationResult:
    """Result of a single question's ReAct loop."""
    interaction_id: str
    query: str
    final_answer: str
    turns_used: int
    messages: List[dict] = field(default_factory=list)
    error: Optional[str] = None


class LLMClient:
    """Async OpenAI client wrapper for multi-turn tool-calling inference."""

    def __init__(self, config: dict):
        self.client = AsyncOpenAI(
            base_url=config["model"]["base_url"],
            api_key=config["model"].get("api_key", "EMPTY"),
        )
        self.model_name = config["model"]["name"]
        self.temperature = config["model"].get("temperature", 0.1)
        self.top_p = config["model"].get("top_p", 0.9)
        self.max_tokens = config["model"].get("max_tokens", 1024)
        self.max_turns = config["inference"]["max_turns"]
        self.timeout = config["inference"].get("timeout_per_turn", 60)

        mock_api_url = config["mock_api"]["base_url"]
        self.tool_executor = ToolExecutor(mock_api_url)

    async def run_react_loop(
        self,
        interaction_id: str,
        query: str,
        domain: str,
        query_time: str,
        system_prompt: str,
    ) -> ConversationResult:
        """
        Run the full ReAct loop for a single question.
        Terminates when:
          1. Model calls the 'answer' tool
          2. Model generates text without tool_calls (finish_reason=stop)
          3. max_turns exceeded
          4. timeout
        """
        tools = get_tools_for_domain(domain)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current Time: {query_time}\nQuestion: {query}"},
        ]

        for turn in range(self.max_turns):
            # Call the model
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
                return ConversationResult(
                    interaction_id=interaction_id,
                    query=query,
                    final_answer="I don't know",
                    turns_used=turn + 1,
                    messages=messages,
                    error="timeout",
                )
            except Exception as e:
                return ConversationResult(
                    interaction_id=interaction_id,
                    query=query,
                    final_answer="I don't know",
                    turns_used=turn + 1,
                    messages=messages,
                    error=f"{type(e).__name__}: {e}",
                )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Append assistant message to history
            messages.append(assistant_msg.model_dump())

            # Case 1: No tool calls — model answered directly
            if not assistant_msg.tool_calls:
                content = assistant_msg.content or "I don't know"
                return ConversationResult(
                    interaction_id=interaction_id,
                    query=query,
                    final_answer=content.strip(),
                    turns_used=turn + 1,
                    messages=messages,
                )

            # Case 2: Process tool calls
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # If it's the answer tool, we're done
                if fn_name == "answer":
                    final_ans = fn_args.get("answer", "I don't know")
                    return ConversationResult(
                        interaction_id=interaction_id,
                        query=query,
                        final_answer=final_ans.strip(),
                        turns_used=turn + 1,
                        messages=messages,
                    )

                # Execute the tool in a thread (CRAG client is synchronous)
                result_str = await asyncio.to_thread(
                    self.tool_executor.execute, fn_name, fn_args
                )

                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # Exhausted max turns
        return ConversationResult(
            interaction_id=interaction_id,
            query=query,
            final_answer="I don't know",
            turns_used=self.max_turns,
            messages=messages,
            error="max_turns_exceeded",
        )
