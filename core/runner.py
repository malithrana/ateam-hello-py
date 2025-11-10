import yaml
from collections.abc import Callable
from typing import Any

from core.dataset import make_dataset
from core.evaluation import kmeans_expected_checker
from core.prompt import PROMPT
from core.tools import TOOL_HANDLERS, TOOLS, set_in_namespace, reset_namespace
from core.agent import run_agent_loop

# Single test run
async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[Any],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer_checker: Callable[[Any], bool],
    max_steps: int,
    max_tokens: int,
    model: str,
    dataset: list[Any],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    reset_namespace() # Reset namespace for each run
    set_in_namespace("dataset", dataset) # Re-inject dataset

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=max_steps,
        max_tokens=max_tokens,
        model=model,
        verbose=verbose,
    )

    success = expected_answer_checker(result) if result is not None else False

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}")

    return run_id, success, result


# Main
async def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_runs = config["num_runs"]
    max_steps = config["max_steps"]
    max_tokens = config["max_tokens"]
    model = config["model"]
    concurrent = config["concurrent"]
    num_clusters = config["num_clusters"]
    num_iterations = config["num_iterations"]

    dataset = make_dataset()
    prompt = PROMPT.format(num_clusters=num_clusters, num_iterations=num_iterations)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            expected_answer_checker=lambda answer: kmeans_expected_checker(
                answer, num_clusters, dataset
            ),
            max_steps=max_steps,
            max_tokens=max_tokens,
            model=model,
            dataset=dataset,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
    else:
        results = [await task for task in tasks]

    successes = sum(success for _, success, _ in results)
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{ '=' * 60}")