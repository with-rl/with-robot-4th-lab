from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local src directory is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph.runner import PlannerRunner  # type: ignore  # noqa: E402
from common.logger import get_logger  # type: ignore  # noqa: E402

logger = get_logger(__name__, is_save=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLDT robot planner.")
    parser.add_argument("user_query", help="User command to decompose.")
    parser.add_argument(
        "--context",
        help="Optional environment context passed to Task-level node.",
        default=None,
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (defaults to configs/config.yaml).",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = PlannerRunner(config_path=args.config)
    result = runner.run(args.user_query, context=args.context)
    logger.info("Planner completed with %d primitive actions.", len(result["actions"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
