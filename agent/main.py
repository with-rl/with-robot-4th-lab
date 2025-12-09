from __future__ import annotations

import argparse
import time

import requests
from dotenv import load_dotenv

from agent.src.config.config_decomp import load_config
from agent.src.runner.executor import TaskExecutor
from agent.src.runner.runner import DecompRunner
from agent.src.runner.state import BaseStateMaker

url = "http://127.0.0.1:8800"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLDT robot planner.")
    parser.add_argument("user_query", help="User command to decompose.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()

    config = load_config("./agent/src/config/config_decomp.yaml")
    state_maker = BaseStateMaker(config)
    print(config)

    user_query = args.user_query

    state = state_maker.make(user_query=user_query)

    runner = DecompRunner(config=config)
    final_state = runner.invoke(state)

    task_outputs = final_state["tasks"]["task_outputs"]

    task_executor = TaskExecutor()
    task_executor.execute(task_outputs)
