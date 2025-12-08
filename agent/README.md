# Robot Agent - Supervised Hierarchical Planning System

A LangGraph-based robot planning agent that decomposes natural language commands into executable robot tasks through a supervised, multi-stage planning pipeline with interactive feedback loops.

## 📋 Overview

The robot agent uses a **supervised planning approach** that combines:
- **Intent Classification** - Determines request type (new task, modification, question, end)
- **Feasibility Checking** - Validates whether requests can be executed
- **Interactive Feedback** - Provides explanations when requests cannot be fulfilled
- **Question Answering** - Handles user queries about environment or capabilities
- **Hierarchical Decomposition** - Breaks down feasible tasks into goals and executable tasks

## 🏗️ System Architecture

### Graph Structure

![Planning Graph](graph.png)

The system implements a **StateGraph** with the following nodes and conditional routing:

```
START → user_input → intent → [Router: intent]
                                   ├─→ end (END)
                                   ├─→ accept → supervisor → [Router: supervisor]
                                   ├─→ accept_no_feedback → feedback → user_input
                                   ├─→ new → supervisor                            │
                                   └─→ question → question_answer → user_input    │
                                                                                   │
supervisor router: ──────────────────────────────────────────────────────────────┘
    ├─→ feasible → goal_decomp → task_decomp → END
    └─→ not_feasible → feedback → user_input
```

### Node Descriptions

| Node | Purpose | Output |
|------|---------|--------|
| **user_input** | Captures user query interactively | Adds query to state |
| **intent** | Classifies user's intention | `{intent, reason, needs_feedback}` |
| **supervisor** | Validates feasibility with robot capabilities | `{feasible, reason}` |
| **feedback** | Generates explanation for infeasible/unclear requests | `{feedback_message}` |
| **question_answer** | Answers environment/capability questions | `{question, answer}` |
| **goal_decomp** | Decomposes command into high-level subgoals | `{subgoals: [str]}` |
| **task_decomp** | Converts subgoals into executable task sequences | `{tasks: [dict]}` |

### Routing Logic

#### Intent Router
Routes based on classified user intention:
- `"end"` → Terminate conversation (END)
- `"accept"` → Accept modified request → supervisor
- `"accept_no_feedback"` → Accept with feedback → feedback
- `"new"` → New task request → supervisor
- `"question"` → User question → question_answer

#### Supervisor Router
Routes based on feasibility check:
- `"feasible"` → Proceed to planning → goal_decomp
- `"not_feasible"` → Provide feedback → feedback

## 📁 Project Structure

```
robot_agent/
├── main.py                    # Entry point for CLI execution
├── graph.png                  # System architecture diagram
├── environment.yml            # Conda environment specification
├── src/
│   ├── common/
│   │   ├── enums.py          # Model name enumerations (GPT-4, GPT-5 variants)
│   │   ├── errors.py         # Custom exception hierarchy
│   │   └── logger.py         # Centralized logging with rotation
│   ├── config/
│   │   ├── config.py         # Pydantic configuration loader
│   │   └── config.yaml       # Node settings, skills, task templates
│   ├── prompts/
│   │   ├── planning_prompt.py    # Goal/task decomposition prompts
│   │   └── process_prompt.py     # Intent/supervisor/feedback prompts
│   ├── runner/
│   │   ├── state.py          # StateSchema and StateMaker
│   │   ├── graph.py          # LLM chain builders and graph constructor
│   │   ├── runner.py         # SupervisedPlanRunner orchestration
│   │   └── text.py           # Formatters for objects/skills/groups
│   ├── utils/
│   │   └── file.py           # File I/O utilities (json, yaml, pkl, csv)
│   ├── rag/                  # (Placeholder for retrieval)
│   └── tools/                # (Placeholder for external tools)
├── data/                      # Runtime data storage
└── test_planning.ipynb        # Interactive testing notebook
```

## 🔄 Data Flow

### State Schema
```python
StateSchema = {
    "user_queries": List[str],              # User input history
    "inputs": Dict[str, Any],               # Environment context (objects, skills, groups)
    "intent_result": Dict[str, Any],        # Intent classification output
    "supervisor_result": Dict[str, Any],    # Feasibility check output
    "feedback_result": Dict[str, Any],      # Generated feedback message
    "feedback_loop_count": int,             # Number of feedback iterations
    "subgoals": List[str],                  # High-level goal decomposition
    "tasks": List[Dict[str, Any]],          # Executable task sequences
    "question_answers": List[Dict[str, Any]] # Q&A history
}
```

### Execution Flow Example

**User Input:** *"Bring the apple to the table"*

1. **user_input** → Captures: `"Bring the apple to the table"`
2. **intent** → Classifies: `{intent: "new", reason: "User wants robot to perform new task", needs_feedback: false}`
3. **supervisor** → Validates: `{feasible: true, reason: "Robot has GoToObject, PickObject, PlaceObject skills"}`
4. **goal_decomp** → Decomposes: `{subgoals: ["Bring the apple to the table"]}`
5. **task_decomp** → Plans:
   ```json
   {
     "tasks": [
       {"skill": "GoToObject", "target": "apple"},
       {"skill": "PickObject", "target": "apple"},
       {"skill": "GoToObject", "target": "table"},
       {"skill": "PlaceObject", "target": "table", "object": "apple"}
     ]
   }
   ```

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate robot_agent

# Or use pip
pip install -r requirements.txt
```

### Configuration

Edit `src/config/config.yaml`:

```yaml
runner:
  intent_node:
    model_name: gpt41mini           # OpenAI model for intent classification
    prompt_cache_key: intent_node   # Cache key for prompt optimization
  supervisor_node:
    model_name: gpt41mini
    prompt_cache_key: supervisor_node
  # ... (other nodes)

skills:
  - name: robot1
    skills: ['GoToObject', 'OpenObject', 'CloseObject', 'PickObject', 'PlaceObject']

tasks:
  GoToObject:
    description: "Move to the specified object."
    template: "GoToObject <robot><object>"
  # ... (other task templates)
```

### Usage

```python
from src.config.config import load_config
from src.runner.state import StateMaker
from src.runner.runner import SupervisedPlanRunner

# Load configuration
config = load_config()

# Create state factory
state_maker = StateMaker(config, url="http://127.0.0.1:8800")

# Initialize runner
runner = SupervisedPlanRunner(config)

# Run planning pipeline
initial_state = state_maker.make(user_query="Bring me a cup")
final_state = runner.invoke(initial_state)

# Access results
print(final_state["subgoals"])
print(final_state["tasks"])
```

### CLI Execution

```bash
python main.py "Bring the apple to the table"
```

## 🛠️ Key Components

### LLM Chain Architecture

Each node is built using `make_normal_node()`:

```python
make_normal_node(
    llm=create_llm(model_name, temperature, prompt_cache_key),
    prompt_text=PROMPT_TEMPLATE,
    make_inputs=input_formatter_function,
    parser_output=PydanticOutputModel,
    state_key="result_field",
    state_append=False,
    node_name="NODE_NAME"
)
```

**Features:**
- Automatic Pydantic output parsing
- Format instruction injection
- Token usage tracking
- Rate limit handling with exponential backoff
- Model name resolution and tagging

### Error Handling

Custom error hierarchy with structured context:

```python
class BaseServiceError(Exception):
    error_code: str
    status_code: int
    domain: str
    details: Dict[str, Any]
```

**Error Types:**
- `ConfigError` - Invalid configuration
- `LLMError` - API call failures
- `RateLimitExceededError` - Rate limit violations
- `GraphExecutionError` - Pipeline failures
- `ParsingError` - Output parsing issues

### Logging

Automatic file rotation with module-level loggers:

```python
from src.common.logger import get_logger

logger = get_logger(__name__, is_save=True)
logger.info("Processing started")
logger.error("Failed to parse output", exc_info=True)
```

## 🔧 Advanced Features

### Prompt Caching

OpenAI prompt caching reduces costs for repeated calls:

```yaml
runner:
  intent_node:
    prompt_cache_key: intent_node  # Enables caching for this node
```

### LLM Instance Caching

Runner maintains a cache to avoid recreating models:

```python
cache_key = (model_name, temperature, prompt_cache_key, bind_tools)
llm = self._llm_cache.get(cache_key) or create_llm(...)
```

### Environment Integration

Fetches live environment data via HTTP:

```python
state_maker = StateMaker(config, url="http://127.0.0.1:8800")
inputs = state_maker.make_inputs()
# Returns: {object_text, skill_text, group_list_text}
```

## 📊 Monitoring

### Token Usage Tracking

Each LLM call records:
- `total_tokens` - Total tokens consumed
- `x-ratelimit-remaining-tokens` - Remaining quota
- `x-ratelimit-remaining-requests` - Remaining request count

### Callback Support

```python
runner = SupervisedPlanRunner(
    config,
    token_information_changed_callback=lambda info: print(info)
)
```

## 🧪 Testing

```bash
# Interactive testing
jupyter notebook test_planning.ipynb

# Unit tests (if available)
pytest tests/
```

## 🎯 Design Principles

1. **Separation of Concerns** - Intent, feasibility, and planning are distinct stages
2. **User-Centric** - Interactive feedback loop ensures clarity
3. **Flexibility** - YAML configuration for easy model/skill updates
4. **Robustness** - Comprehensive error handling and retry logic
5. **Observability** - Detailed logging and token tracking

## 📝 Example Scenarios

### Scenario 1: Feasible Request
```
User: "Pick up the fork from the counter"
Intent: new → Supervisor: feasible → Goal Decomp → Task Decomp → END
```

### Scenario 2: Infeasible Request
```
User: "Fly to the ceiling"
Intent: new → Supervisor: not_feasible → Feedback: "Robot cannot fly..." → User Input
```

### Scenario 3: Question
```
User: "What objects are on the table?"
Intent: question → Question Answer: "bowl, fork, plate" → User Input
```

### Scenario 4: Request Modification
```
User: "Actually, bring two apples"
Intent: accept → Supervisor: feasible → Goal Decomp → ...
```

## 🔮 Future Enhancements

- [ ] Action-level planning with primitive motions
- [ ] RAG integration for knowledge retrieval
- [ ] Multi-robot coordination
- [ ] Visual grounding with object detection
- [ ] Execution monitoring and replanning
- [ ] Natural language plan explanations

## 📚 Dependencies

| Library | Purpose |
|---------|---------|
| `langchain` | LLM orchestration framework |
| `langgraph` | Graph-based workflow management |
| `openai` | GPT model API access |
| `pydantic` | Data validation and settings |
| `pyyaml` | Configuration file parsing |

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

Based on MLDT (Multi-Level Decomposition Task) planning architecture with supervised interaction capabilities.
