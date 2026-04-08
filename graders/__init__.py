from graders.base_grader import BaseGrader
from graders.grader_easy import EasyGrader
from graders.grader_medium import MediumGrader
from graders.grader_hard import HardGrader


def get_grader(task: str) -> BaseGrader:
    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }
    if task not in graders:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {list(graders)}")
    return graders[task]


__all__ = ["get_grader", "BaseGrader", "EasyGrader", "MediumGrader", "HardGrader"]
