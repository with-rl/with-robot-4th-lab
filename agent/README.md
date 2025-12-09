# Robot Agent – Goal/Task Decomposition

Two-stage LangGraph agent that decomposes a user command into subgoals and executable tasks. The **actual end-to-end execution guide lives in `../test_planning.ipynb`**—follow that notebook for a working run of the code in `agent/src`.

## Overview
- Two nodes: goal decomposition → task decomposition.
- Uses OpenAI-compatible LLMs (`langgraph`, `langchain`) with Pydantic parsing.
- Pulls environment objects from a local simulator API (`http://127.0.0.1:8800/env`) and formats available robot skills from config.
- Optional executor sends HTTP actions to the same simulator.

## Architecture
```
START
  ↓
goal_decomp    # Break user query into subgoals using object context
  ↓
task_decomp    # Turn each subgoal into skill-level task lists
  ↓
END
```

| Node | Purpose | Output key |
|------|---------|------------|
| `goal_decomp` | Attribute-aware subgoal splitter based on `object_text` and the latest `user_query`. | `subgoals` |
| `task_decomp` | Maps each subgoal to ordered tasks that only use skills defined in config. | `tasks` |

State fields (from `BaseStateSchema`): `user_queries`, `inputs` (`object_text`, `skill_text`), `subgoals`, `tasks`.

## Project Structure
```
agent/
├── main.py
├── environment.yml
├── src/
│   ├── common/           # enums, errors, logger
│   ├── config/
│   │   ├── config_decomp.py/.yaml   # default config used in the notebook run
│   │   └── config_full.py/.yaml     # extended config (intent/supervisor)
│   ├── prompts/          # goal/task prompt templates
│   ├── runner/           # state maker, graph wiring, executor
│   ├── utils/            # file helpers
│   ├── rag/, tools/      # placeholders
├── data/
└── ../test_planning.ipynb    # canonical walkthrough for running the planner
```

## Execution (Notebook-first)
The notebook is the authoritative, working path for running the agent.

1) Install deps (conda or pip):
```bash
conda env create -f environment.yml
conda activate robot_agent
# or: pip install -r requirements.txt
```
2) Ensure the local simulator API is up at `http://127.0.0.1:8800` (the notebook calls `/env` and `/send_action`).
3) Launch the notebook from repo root:
```bash
jupyter notebook test_planning.ipynb
```
4) Run cells in order (they already mirror the code in `agent/src`):
   - Fetch environment objects: `requests.get("http://127.0.0.1:8800/env")`
   - Load config: `load_config("./agent/src/config/config_decomp.yaml")`
   - Build state: `BaseStateMaker(config).make(user_query="...")`
   - Plan: `DecompRunner(config).invoke(state)`
   - Inspect tasks: `final_state["tasks"]["task_outputs"]`
   - (Optional) Execute against the simulator: `TaskExecutor().execute(task_outputs)`

Minimal reference snippet (same as the notebook):
```python
from agent.src.config.config_decomp import load_config
from agent.src.runner.state import BaseStateMaker
from agent.src.runner.runner import DecompRunner
from agent.src.runner.executor import TaskExecutor

config = load_config("./agent/src/config/config_decomp.yaml")
state = BaseStateMaker(config).make(
    user_query="Organize the objects to the bowls according to their colors"
)
final_state = DecompRunner(config).invoke(state)
task_outputs = final_state["tasks"]["task_outputs"]
TaskExecutor().execute(task_outputs)  # hits the simulator HTTP API
```

> If you prefer CLI experimentation, `python main.py "<query>"` is available, but the notebook is the maintained path for now.

## Configuration
`agent/src/config/config_decomp.yaml` is the default for the notebook run:
```yaml
paths:
  output_dir: "output/"
  prompt_dir: "src/graph/prompts/"

runner:
  goal_decomp_node:
    model_name: gpt41mini
    prompt_cache_key: goal_decomp_node
  task_decomp_node:
    model_name: gpt41mini
    prompt_cache_key: task_decomp_node

skills:
  - name: robot1
    skills: ['GoToObject', 'PickObject', 'PlaceObject']
```
`config_full.yaml` adds intent/supervisor nodes but is not exercised in `test_planning.ipynb`.

## Logging
Module-level loggers live in `src/common/logger.py`. Use `get_logger(__name__, is_save=True)` for rotating file output.

## Dependencies
| Library | Purpose |
|---------|---------|
| `langchain` / `langgraph` | LLM orchestration and workflow graph |
| `openai` | Chat completion models |
| `pydantic` | Config + output validation |
| `requests` | Environment/simulator HTTP calls |

## License
See `LICENSE`.
