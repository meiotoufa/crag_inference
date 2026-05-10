"""
toolbench_tools.py - Dynamic tool definitions and execution for ToolBench.

Unlike CRAG (fixed tools per domain), ToolBench provides per-query API lists.
Each query specifies which tools are available via its `api_list` field.
"""
import json
import hashlib
from typing import Any, Dict, List


# =============================================================================
# ANSWER TOOL (universal — signals end of ReAct loop)
# =============================================================================

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "Finish",
        "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart.",
        "parameters": {
            "type": "object",
            "properties": {
                "return_type": {
                    "type": "string",
                    "enum": ["give_answer", "give_up_and_restart"],
                    "description": "'give_answer' if you have the final answer, 'give_up_and_restart' if you cannot proceed."
                },
                "final_answer": {
                    "type": "string",
                    "description": "The final answer to the task. Required if return_type is 'give_answer'."
                }
            },
            "required": ["return_type"]
        }
    }
}


# =============================================================================
# DYNAMIC TOOL CONSTRUCTION
# =============================================================================

def build_tool_schema(api_info: dict) -> dict:
    """
    Convert a ToolBench API entry to OpenAI function-calling schema.

    ToolBench API entry format:
    {
        "category_name": "...",
        "tool_name": "...",
        "api_name": "...",
        "api_description": "...",
        "required_parameters": [...],
        "optional_parameters": [...],
        "method": "GET",
        "template_response": {...}
    }
    """
    tool_name = api_info.get("tool_name", "unknown_tool")
    api_name = api_info.get("api_name", "unknown_api")
    description = api_info.get("api_description", "No description available.")

    # Build unique function name: tool_name__api_name (sanitized)
    fn_name = f"{tool_name}__{api_name}".replace(" ", "_").replace("-", "_").replace("/", "_")
    # Ensure name is valid (alphanumeric + underscore, max 64 chars)
    fn_name = "".join(c if c.isalnum() or c == "_" else "_" for c in fn_name)[:64]

    # Build parameters
    properties = {}
    required = []

    for param in api_info.get("required_parameters", []):
        param_name = param.get("name", "param")
        properties[param_name] = {
            "type": _map_type(param.get("type", "STRING")),
            "description": param.get("description", f"Parameter: {param_name}"),
        }
        if param.get("default") is not None:
            properties[param_name]["default"] = param["default"]
        required.append(param_name)

    for param in api_info.get("optional_parameters", []):
        param_name = param.get("name", "param")
        properties[param_name] = {
            "type": _map_type(param.get("type", "STRING")),
            "description": param.get("description", f"Optional parameter: {param_name}"),
        }
        if param.get("default") is not None:
            properties[param_name]["default"] = param["default"]

    return {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": f"[{tool_name}] {description}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    }


def _map_type(toolbench_type: str) -> str:
    """Map ToolBench parameter types to JSON Schema types."""
    mapping = {
        "STRING": "string",
        "NUMBER": "number",
        "INTEGER": "integer",
        "BOOLEAN": "boolean",
        "ENUM": "string",
    }
    return mapping.get(toolbench_type.upper(), "string")


def get_tools_for_query(api_list: List[dict]) -> List[dict]:
    """Build OpenAI tool schemas from a query's api_list + Finish tool."""
    tools = []
    for api_info in api_list:
        tools.append(build_tool_schema(api_info))
    tools.append(ANSWER_TOOL)
    return tools


# =============================================================================
# TOOL EXECUTOR (SIMULATED)
# =============================================================================

MAX_RESULT_LENGTH = 4000


class ToolBenchExecutor:
    """
    Executes ToolBench tool calls.

    In simulated mode: returns template_response or a generic success message.
    In real mode: calls RapidAPI (requires API key).
    """

    def __init__(self, config: dict):
        self.use_simulated = config.get("tool_server", {}).get("use_simulated", True)
        self.rapidapi_key = config.get("tool_server", {}).get("rapidapi_key", "")
        self._api_registry: Dict[str, dict] = {}

    def register_apis(self, api_list: List[dict]):
        """Register the APIs available for the current query."""
        self._api_registry.clear()
        for api_info in api_list:
            tool_name = api_info.get("tool_name", "unknown_tool")
            api_name = api_info.get("api_name", "unknown_api")
            fn_name = f"{tool_name}__{api_name}".replace(" ", "_").replace("-", "_").replace("/", "_")
            fn_name = "".join(c if c.isalnum() or c == "_" else "_" for c in fn_name)[:64]
            self._api_registry[fn_name] = api_info

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return result as string."""
        if tool_name == "Finish":
            return_type = arguments.get("return_type", "give_answer")
            if return_type == "give_answer":
                return arguments.get("final_answer", "")
            else:
                return '{"status": "restart_requested"}'

        if self.use_simulated:
            return self._execute_simulated(tool_name, arguments)
        else:
            return self._execute_real(tool_name, arguments)

    def _execute_simulated(self, tool_name: str, arguments: dict) -> str:
        """Return simulated response based on template_response or generated mock."""
        api_info = self._api_registry.get(tool_name)
        if not api_info:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        # Use template_response if available
        template = api_info.get("template_response")
        if template:
            result_str = json.dumps(template, ensure_ascii=False)
        else:
            # Generate a deterministic mock response
            seed = hashlib.md5(f"{tool_name}_{json.dumps(arguments, sort_keys=True)}".encode()).hexdigest()[:8]
            result_str = json.dumps({
                "status": "success",
                "data": f"Simulated response for {tool_name} (seed={seed})",
                "parameters_received": arguments,
            }, ensure_ascii=False)

        if len(result_str) > MAX_RESULT_LENGTH:
            result_str = result_str[:MAX_RESULT_LENGTH] + "... [truncated]"
        return result_str

    def _execute_real(self, tool_name: str, arguments: dict) -> str:
        """Call real RapidAPI endpoint."""
        import requests

        api_info = self._api_registry.get(tool_name)
        if not api_info:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        # Build RapidAPI request
        host = api_info.get("tool_name", "").lower().replace(" ", "-")
        url = f"https://{host}.p.rapidapi.com/{api_info.get('api_name', '')}"
        method = api_info.get("method", "GET").upper()

        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": f"{host}.p.rapidapi.com",
        }

        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, params=arguments, timeout=30)
            else:
                resp = requests.post(url, headers=headers, json=arguments, timeout=30)

            result_str = resp.text
            if len(result_str) > MAX_RESULT_LENGTH:
                result_str = result_str[:MAX_RESULT_LENGTH] + "... [truncated]"
            return result_str
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    # Verify schema generation
    sample_api = {
        "category_name": "Finance",
        "tool_name": "Currency Exchange",
        "api_name": "listquotes",
        "api_description": "List available currency quotes.",
        "required_parameters": [],
        "optional_parameters": [
            {"name": "format", "type": "STRING", "description": "Response format", "default": "json"}
        ],
        "method": "GET",
        "template_response": {"quotes": ["USD", "EUR", "GBP"]},
    }
    schema = build_tool_schema(sample_api)
    assert schema["function"]["name"] == "Currency_Exchange__listquotes"
    assert "parameters" in schema["function"]

    tools = get_tools_for_query([sample_api])
    assert len(tools) == 2  # 1 API + Finish
    assert tools[-1]["function"]["name"] == "Finish"

    executor = ToolBenchExecutor({"tool_server": {"use_simulated": True}})
    executor.register_apis([sample_api])
    result = executor.execute("Currency_Exchange__listquotes", {"format": "json"})
    parsed = json.loads(result)
    assert "quotes" in parsed

    print("ok")
