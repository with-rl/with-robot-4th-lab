"""Planner runner wiring the LangGraph Goal -> Task -> Action pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from ..common.enums import ModelNames
from ..common.errors import GraphInitializeError
from ..common.logger import get_logger
from ..prompts import planning_prompt
from . import graph as graph_module
from .state import BaseStateSchema, SupervisedPlanStateSchema

logger = get_logger(__name__)


class GraphRunner:
    def __init__(
        self,
        config,
        token_information_changed_callback: Callable | None = None,
    ):
        self.config = config
        self.graph: Any | None = None
        self.graph_config: Dict[str, Any] | None = None
        self.retriever = None

        self._llm_cache: Dict[Tuple[str, float, str | None, bool], Any] = {}

        if token_information_changed_callback is not None:
            self.token_information_changed_callback = token_information_changed_callback
        else:
            self.token_information_changed_callback = None

    def set_retriever(self, retriever):
        self.retriever = retriever
        self.graph = None
        self.graph_config = None

    def build_graph(self):
        raise NotImplementedError("Subclasses must implement build_graph().")

    def _get_llm(
        self,
        model_name: ModelNames | str,
        *,
        temperature: float = 0.0,
        prompt_cache_key: str | None = None,
        bind_tools: bool = False,
    ):
        if isinstance(model_name, ModelNames):
            model_enum = model_name
        else:
            try:
                # Try interpret as enum value (e.g., "gpt-4.1")
                model_enum = ModelNames(model_name)
            except ValueError:
                # Fallback to enum key (e.g., "gpt_4_1")
                model_enum = ModelNames[model_name]

        cache_key = (model_enum.value, temperature, prompt_cache_key, bind_tools)
        llm = self._llm_cache.get(cache_key)
        if llm is None:
            llm = graph_module.create_llm(
                model_name=model_enum,
                temperature=temperature,
                prompt_cache_key=prompt_cache_key,
            )
            self._llm_cache[cache_key] = llm
        return llm

    def _ensure_graph(self) -> Tuple[Any, Dict[str, Any]]:
        if self.graph is None or self.graph_config is None:
            self.graph, self.graph_config = self.build_graph()
        if self.graph is None or self.graph_config is None:
            raise GraphInitializeError(
                "Graph or graph_config is not properly initialized."
            )
        return self.graph, self.graph_config

    def invoke(self, state):
        graph, graph_config = self._ensure_graph()
        final_state = graph.invoke(state, graph_config)
        return final_state

    def batch(self, states):
        graph, graph_config = self._ensure_graph()
        final_states = graph.batch(states, graph_config)
        return final_states


class DecompRunner(GraphRunner):
    def build_graph(self):
        nodes = {}
        routers = {}
        nodes["goal_decomp"] = graph_module.make_normal_node(
            llm=self._get_llm(
                model_name=self.config.runner.goal_decomp_node.model_name,
                prompt_cache_key=self.config.runner.goal_decomp_node.prompt_cache_key,
            ),
            prompt_text=planning_prompt.GOAL_DECOMP_NODE_PROMPT,
            make_inputs=planning_prompt.make_goal_decomp_node_inputs,
            parser_output=planning_prompt.GoalDecompNodeParser,
            state_key="subgoals",
            state_append=False,
            node_name="GOAL_DECOMP_NODE",
        )
        nodes["task_decomp"] = graph_module.make_normal_node(
            llm=self._get_llm(
                model_name=self.config.runner.task_decomp_node.model_name,
                prompt_cache_key=self.config.runner.task_decomp_node.prompt_cache_key,
            ),
            prompt_text=planning_prompt.TASK_DECOMP_NODE_PROMPT,
            make_inputs=planning_prompt.make_task_decomp_node_inputs,
            parser_output=planning_prompt.TaskDecompNodeParser,
            state_key="tasks",
            state_append=False,
            node_name="TASK_DECOMP_NODE",
        )
        return graph_module.make_decomp_plan_graph(
            state_schema=BaseStateSchema,
            nodes=nodes,
            thread_id="decomp_planning",
        )
