from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


# --- Enums ---

class Friction(str, Enum):
    low = "low"
    high = "high"


class Alignment(str, Enum):
    aligned = "aligned"
    misaligned = "misaligned"


class Stiffness(str, Enum):
    rigid = "rigid"
    compliant = "compliant"


# --- Core models ---

class HiddenState(BaseModel):
    friction_level: Friction
    alignment_error: Alignment
    stiffness: Stiffness


class Observation(BaseModel):
    position: float
    force_feedback: float
    lateral_instability: float
    progress: float
    failure_signal: str
    last_action: str
    step_count: int

    @field_validator("position", "force_feedback", "lateral_instability", "progress")
    @classmethod
    def must_be_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            # Clamp instead of raise to be more robust during simulation
            return max(0.0, min(1.0, v))
        return v

    @field_validator("failure_signal")
    @classmethod
    def must_be_valid_signal(cls, v: str) -> str:
        valid = {"none", "jam", "slip", "unstable"}
        if v not in valid:
            raise ValueError(f"failure_signal must be one of {valid}, got {v!r}")
        return v


class R2EAction(BaseModel):
    action: str


class RewardComponents(BaseModel):
    progress_reward: float = 0.0
    step_penalty: float = 0.0
    invalid_penalty: float = 0.0
    probe_reward: float = 0.0
    failure_penalty: float = 0.0
    success_reward: float = 0.0

    def total(self) -> float:
        return (
            self.progress_reward
            + self.step_penalty
            + self.invalid_penalty
            + self.probe_reward
            + self.failure_penalty
            + self.success_reward
        )


class GradeResult(BaseModel):
    score: float
    success: float
    efficiency: float
    correctness: float
    details: dict[str, Any]


class StepRecord(BaseModel):
    step: int
    action: str
    reward: float
    done: bool
    observation: Observation


class EpisodeLog(BaseModel):
    task: str
    seed: int
    steps: list[StepRecord]
    final_obs: Observation
    total_reward: float
    done: bool
    revealed_probes: list[str]
    hidden_state: HiddenState


class TaskConfig(BaseModel):
    name: str
    active_variables: list[str]
    deceptive_reward_enabled: bool
    max_steps: int
    seed_range: tuple[int, int]
