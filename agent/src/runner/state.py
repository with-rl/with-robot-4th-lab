"""Planner state definitions and helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import requests
from typing_extensions import TypedDict

from ..config.config_decomp import Config, RobotSkillConfig


def make_object_text(url):
    response = requests.get(f"{url}/env")
    all = response.json()
    objects = all["objects"]
    print(objects)
    total_object_text = "{{\n"
    for obj in objects:
        object_text = f'"object_name": "{obj}",\n'
        total_object_text += object_text

    total_object_text += "}}"

    return total_object_text


def make_skill_text(config_skills: list[RobotSkillConfig]) -> str:
    skill_text_list = []
    for robot_skill in config_skills:

        skill_text = f"from {robot_skill.name}.skills import "
        for skill in robot_skill.skills:
            skill_text += f"{skill}"
            if skill != robot_skill.skills[-1]:
                skill_text += ", "
        skill_text_list.append(skill_text)

    return "\n".join(skill_text_list)


class BaseStateSchema(TypedDict, total=False):
    """State contract for the planner LangGraph workflow."""

    user_queries: List[str]
    inputs: Dict[str, Any]
    subgoals: List[str]
    tasks: List[Dict[str, Any]]


class SupervisedPlanStateSchema(TypedDict, total=False):
    user_queries: List[str]
    inputs: Dict[str, Any]
    subgoals: List[str]
    tasks: List[Dict[str, Any]]
    intent_result: Dict[str, Any]
    supervisor_result: Dict[str, Any]
    feedback_result: Dict[str, Any]
    feedback_loop_count: int
    question_answers: List[Dict[str, Any]]


class BaseStateMaker:
    """Factory for creating planner state inputs."""

    def __init__(self, config: Config, url: None | str = None) -> None:
        self.config = config
        if url is not None:
            self.url = url
        else:
            self.url = "http://127.0.0.1:8800"

        self.base_state = self._make_base_state()

    def _make_base_state(self):
        return {
            "user_queries": [],
            "inputs": {},
            "subgoals": [],
            "tasks": [],
        }

    def _make_inputs(self):
        inputs = {}
        print("Making inputs for state...")
        inputs["object_text"] = make_object_text(self.url)
        inputs["skill_text"] = make_skill_text(self.config.skills)
        print(f"url: {self.url}")
        return inputs

    def make(self, *, user_query: str):
        """Create a fresh state with defaults."""

        state = copy.deepcopy(self.base_state)
        state["user_queries"] = [user_query]
        state["inputs"] = self._make_inputs()
        return state


class SupervisedPlanStateMaker(BaseStateMaker):
    """Factory for creating supervised plan state inputs."""

    def _make_base_state(self):
        return {
            "user_queries": [],
            "inputs": {},
            "intent_result": {},
            "supervisor_result": {},
            "feedback_result": {},
            "feedback_loop_count": 0,
            "subgoals": [],
            "tasks": [],
            "question_answers": [],
        }
