hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.


# RL Task Documentation: k-means clustering without telling the model it is k-means clustering

## 1. Task Overview

This RL task is designed to get an LLM to perform unsupervised clustering on 3D data using the K-Means algorithm. We do not explicitly say it is k-means. Instead we provide steps of the algorithm so that the agent can determine on its own to take actions.

The model is asked to:

Normalize a dataset of 3D points,

Perform clustering with three clusters and three iterations (you can change this in ```config.yaml```),

Submit a valid cluster assignment mapping each original (unnormalized) point to its cluster index.

The task reflects the kind of work an ML engineer or researcher might do—implementing and validating unsupervised learning techniques.

## 2. Motivation

This task teaches the model how to:

Apply data normalization (mean subtraction and standard deviation scaling),

Execute iterative optimization procedures (centroid updates),

Implement Euclidean distance-based clustering, and

Produce structured, verifiable outputs for ML workflows.

It also challenges the model’s ability to reason numerically, organize data, and apply geometric intuition key skills in applied ML research.

## 3. Project Structure

The codebase is organized into a `core` directory for all Python source files, with a minimal `main.py` at the root serving as the application's entry point.

**Root Directory:**
- `main.py`: The application's entry point, which imports and runs the `main` function from `core.runner`.
- `config.yaml`: Configuration file specifying experiment parameters.
- `README.md`: This documentation file.
- `log_output.txt`: Output log from test runs.

**core/ Directory:**
- `agent.py`: Contains the core RL agent loop (`run_agent_loop`) managing message exchange with the Claude API.
- `runner.py`: Orchestrates the test runs, including `run_single_test()` (executes a single test run with the agent) and `main()` (loads config, creates dataset, runs multiple test iterations, and reports results).
- `dataset.py`: Data generation, including `make_dataset()` (generates 50 3D points from three Gaussian distributions with fixed seed (42) for reproducibility).
- `evaluation.py`: Answer validation, including `kmeans_expected_checker()` (validates format, completeness, and geometric correctness of submitted cluster assignments).
- `prompt.py`: Task instruction template, including `PROMPT` (natural language instructions provided to the model).
- `tools.py`: Tool definitions and handlers, including `python_expression_tool()` (executes Python code), `submit_answer_tool()` (accepts the final clustering result), `TOOLS` (schema definitions), and `TOOL_HANDLERS` (maps tool names to handler functions).

## 4. Task Components
#### 4.1 Prompt

The model receives a detailed natural-language instruction specifying:

The dataset of 3D points (generated in ```make_dataset()```),

Steps for normalization and clustering,

The expected format for the final answer (a dictionary mapping point strings to cluster IDs).

#### 4.2 Tools

The task defines two tools:

```python_expression``` – Executes arbitrary Python expressions within a controlled namespace for mathematical computation.

```submit_answer``` – Used to submit the final clustering result for evaluation.

#### 4.3 Data

The dataset is composed of 50 3D points drawn from three Gaussian distributions with distinct means and equal covariance.
Each run generates identical data due to a fixed random seed (np.random.seed(42)).

## 5. Grader Function

The grader (```kmeans_expected_checker```) verifies whether the model’s submission is correct by checking:

Format correctness

The answer must be a dictionary mapping stringified points to integer cluster IDs (0, 1, or 2).

All 50 original points must be present.

Cluster validity

Each cluster must contain at least one point.

Normalization and geometric consistency

The grader re-normalizes the data (using the correct mean and standard deviation).

It recalculates centroids in normalized space based on the submitted assignments.

For each normalized point, it ensures the assigned centroid is indeed the closest one by Euclidean distance.

Any violation (e.g., misassigned points or empty clusters) triggers a failure message such as:

“A point is closer to a centroid of another cluster than its own cluster.”

In summary, the grader assesses:

Correct data normalization,

Correct clustering behavior,

Complete and well-formatted output mapping,

Robust adherence to the prompt instructions.

## 6. Failure Analysis and Performance

Let's examine the following output console log.

``` 
$ python main.py 
✓ Run 1: SUCCESS - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
A point is closer to a centroid of another cluster than its own cluster.
✗ Run 2: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
Answer length 54 does not match dataset length 50.
✗ Run 3: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
A point is closer to a centroid of another cluster than its own cluster.
✗ Run 4: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
✗ Run 5: FAILURE - Got None
Submitted points do not match original dataset.
✗ Run 6: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
Failed to parse answer string as JSON.
✗ Run 7: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
A point is closer to a centroid of another cluster than its own cluster.
✗ Run 8: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
A point is closer to a centroid of another cluster than its own cluster.
✗ Run 9: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]
A point is closer to a centroid of another cluster than its own cluster.
✗ Run 10: FAILURE - Got [['(1, 0, 1)', 0], ['(2, 0, 0)', 0], ['(2, 1, -1)', 0],...]

============================================================
Test Results:
  Passed: 1/10
  Failed: 9/10
  Pass Rate: 10.0%
============================================================

```

| Failure Cause                                | Category                              
| -------------------------------------------- | -------------------------------------- 
| Invalid JSON output                          | Output formatting error                
| Wrong cluster mapping                        | Conceptual/logic error                 
| Hit steps limit                              | Procedural error / task incompletion
| Agent adding new data points                 | Logic/reasoning error  

Overall Pass Rate: 10% (1/10 runs)

Multiple modes of failures were noticed due to:

Incorrect centroid updates or point assignments – points were closer to another cluster than their assigned one.

Output formatting issues – invalid JSON structure or missing data.

Incomplete computation – no submission after tool execution (Got None).

Introducing new data points - introducing new data points during clustering

## 7. Conclusion

This RL task successfully demonstrates an agent's ability to implement a fundamental unsupervised learning algorithm—k-means clustering—through natural language instructions and tool-based computation. The task design emphasizes several key competencies:

**Algorithmic reasoning**: The model must understand and execute iterative optimization procedures, including normalization, distance calculations, and centroid updates.

**Numerical precision**: Proper handling of floating-point arithmetic, array operations, and geometric distance metrics is essential for correct clustering.

**Structured output generation**: The model must produce well-formatted, complete answers mapping all points to valid cluster indices.

With a 10% pass rate on the initial configuration (3 iterations, 50 points, 3 clusters), the task reveals that while LLMs can grasp the conceptual framework of K-means, achieving geometric correctness consistently remains challenging. Common failure modes include:

- **Premature convergence**: Insufficient iterations leading to suboptimal cluster assignments
- **Numerical errors**: Mistakes in distance calculations or centroid updates
- **Format compliance issues**: Incorrect JSON serialization or missing data points
- **Hallusionations**: Introducing new data points that did not exist in the orgianl set

The modular codebase architecture—separating data generation, evaluation, prompt design, and orchestration—provides a flexible foundation for future experimentation. Potential improvements include:

- Increasing iteration count to ensure convergence
- Adding validation checkpoints during execution
- Providing clearer examples of expected intermediate outputs
- Testing with different models and dataset configurations

Overall, this task serves as a valuable benchmark for assessing LLM capabilities in numerical computation, algorithm implementation, and structured problem-solving within the ML domain.