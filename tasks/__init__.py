from tasks.task_easy import EASY_CONFIG
from tasks.task_medium import MEDIUM_CONFIG
from tasks.task_hard import HARD_CONFIG
from r2e_env.models import TaskConfig

TASK_CONFIGS: dict[str, TaskConfig] = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}

__all__ = ["TASK_CONFIGS"]
