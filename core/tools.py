from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict


# Tool result types
class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


# Tools
namespace = {}  # Global namespace for exec

def reset_namespace():
    """Clears the shared namespace."""
    namespace.clear()


def set_in_namespace(key: str, value: Any):
    """Sets a value in the shared namespace."""
    namespace[key] = value


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    try:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    return {"answer": answer, "submitted": True}


TOOLS = [
    {
        "name": "python_expression",
        "description": "Evaluates a Python expression. The dataset is available as a variable named 'dataset'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {}},
            "required": ["answer"],
        },
    },
]

TOOL_HANDLERS = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}
