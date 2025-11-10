import asyncio
import json
from collections.abc import Callable
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

async def run_agent_loop(
    prompt: str,
    tools: list[Any],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    max_tokens: int = 1000,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=max_tokens, tools=tools, messages=messages
        )

        assert response.stop_reason in [
            "max_tokens",
            "tool_use",
            "end_turn",
        ], f"unsupported stop_reason {response.stop_reason}"
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {max_tokens}. Increase MAX_TOKENS or simplify task."
            )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                if tool_name in tool_handlers:
                    handler = tool_handlers[tool_name]
                    tool_input = content.input
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print(tool_input["expression"])
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print(result)
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")

    return None