from __future__ import annotations

# import argparse
from dotenv import load_dotenv
from src.config.config_decomp import load_config
from src.runner.executor import TaskExecutor
from src.runner.runner import DecompRunner
from src.runner.state import BaseStateMaker

url = "http://127.0.0.1:8800"


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run MLDT robot planner.")
#     parser.add_argument("user_query", help="User command to decompose.")
#     return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    load_dotenv()

    config = load_config("./src/config/config_decomp.yaml")
    state_maker = BaseStateMaker(config)
    print(config)

    # user_query = args.user_query
    user_query = input("Enter your command for the robot: ")

    state = state_maker.make(user_query=user_query)

    runner = DecompRunner(config=config)
    final_state = runner.invoke(state)

    task_outputs = final_state["tasks"]["task_outputs"]

    task_executor = TaskExecutor()
    task_executor.execute(task_outputs)
