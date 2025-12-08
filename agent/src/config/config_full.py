from asyncio import tasks
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..common.errors import UtilsConfigurationError, UtilsValidationError


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str
    prompt_dir: str


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    prompt_cache_key: str | None = None


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_node: NodeConfig
    supervisor_node: NodeConfig
    feedback_node: NodeConfig
    goal_decomp_node: NodeConfig
    task_decomp_node: NodeConfig
    question_answer_node: NodeConfig


class RobotSkillConfig(BaseModel):
    name: str
    skills: list[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    runner: RunnerConfig
    skills: list[RobotSkillConfig]
    tasks: dict[str, dict[str, Any]]


def load_config(config_path: str | Path | None = None) -> Config:
    if config_path is None:
        # Default to this module's config directory when no path is provided
        config_dir = Path(__file__).resolve().parent
        config_path = config_dir / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    if not isinstance(raw_config, dict):
        raise UtilsConfigurationError(
            "Configuration file must contain a top-level mapping.",
            details={"path": str(config_path)},
        )

    try:
        return Config.model_validate(raw_config)
    except ValidationError as error:
        raise UtilsConfigurationError(
            f"Invalid configuration: {error}",
            details={"path": str(config_path)},
        ) from error
