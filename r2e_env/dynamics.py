import random

from r2e_env.models import (
    Alignment,
    Friction,
    HiddenState,
    Observation,
    RewardComponents,
    Stiffness,
    TaskConfig,
)

VALID_ACTIONS = [
    "insert",
    "adjust_left",
    "adjust_right",
    "increase_force",
    "probe_friction",
    "probe_alignment",
    "probe_stiffness",
    "commit_solution",
]


class DynamicsEngine:
    def sample_hidden_state(self, seed: int, task_config: TaskConfig) -> HiddenState:
        rng = random.Random(seed)

        friction = Friction.low
        alignment = Alignment.aligned
        stiffness = Stiffness.rigid

        if "friction" in task_config.active_variables:
            friction = rng.choice([Friction.low, Friction.high])
        if "alignment" in task_config.active_variables:
            alignment = rng.choice([Alignment.aligned, Alignment.misaligned])
        if "stiffness" in task_config.active_variables:
            stiffness = rng.choice([Stiffness.rigid, Stiffness.compliant])

        return HiddenState(
            friction_level=friction,
            alignment_error=alignment,
            stiffness=stiffness
        )

    def apply_action(
        self,
        action: str,
        hidden: HiddenState,
        obs: Observation,
        revealed_probes: set[str],
        task_config: TaskConfig,
    ) -> tuple[Observation, RewardComponents]:
        rc = RewardComponents()
        next_obs = obs.model_copy(deep=True)
        rc.step_penalty = -0.01

        if action == "insert":
            # Success logic
            if hidden.alignment_error == Alignment.misaligned and next_obs.lateral_instability > 0.3:
                # Still misaligned
                next_obs.failure_signal = "unstable"
                # No progress
            elif hidden.friction_level == Friction.high:
                delta = 0.10
                next_obs.position = min(1.0, obs.position + delta)
                rc.progress_reward = 0.1 * delta
            else:
                delta = 0.20
                next_obs.position = min(1.0, obs.position + delta)
                rc.progress_reward = 0.1 * delta

        elif action in ("adjust_left", "adjust_right"):
            if hidden.alignment_error == Alignment.misaligned:
                next_obs.lateral_instability = max(0.1, obs.lateral_instability - 0.4)
                next_obs.failure_signal = "none"
            else:
                # Wasted action on aligned part
                next_obs.lateral_instability = min(1.0, obs.lateral_instability + 0.1)

        elif action == "increase_force":
            # Deceptive reward: gives immediate progress
            delta = 0.40
            next_obs.position = min(1.0, obs.position + delta)
            rc.progress_reward = 0.1 * delta  # +0.04
            
            # Trap condition: High Friction + Compliant Stiffness + Misaligned (or just the first two)
            if (
                hidden.friction_level == Friction.high
                and hidden.stiffness == Stiffness.compliant
                and task_config.deceptive_reward_enabled
            ):
                next_obs.failure_signal = "jam"
                rc.failure_penalty = -1.0

        elif action == "probe_friction":
            next_obs.force_feedback = 0.9 if hidden.friction_level == Friction.high else 0.2
            if "friction" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("friction")

        elif action == "probe_alignment":
            next_obs.lateral_instability = (
                0.8 if hidden.alignment_error == Alignment.misaligned else 0.1
            )
            if "alignment" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("alignment")

        elif action == "probe_stiffness":
            # Stiffness reveals itself through a lateral spike under probing tension
            if hidden.stiffness == Stiffness.compliant:
                next_obs.lateral_instability = min(1.0, obs.lateral_instability + 0.4)
            else:
                next_obs.lateral_instability = max(0.0, obs.lateral_instability - 0.1)
                
            if "stiffness" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("stiffness")

        elif action == "commit_solution":
            if next_obs.position >= 1.0 and next_obs.failure_signal == "none":
                rc.success_reward = 1.0
            else:
                rc.failure_penalty = -1.0

        next_obs.last_action = action
        next_obs.step_count = obs.step_count + 1
        next_obs.progress = next_obs.position

        return (next_obs, rc)

    def is_terminal(self, obs: Observation) -> tuple[bool, str]:
        if obs.failure_signal == "jam":
            return (True, "jam")
        if obs.failure_signal == "slip":
            return (True, "slip")
        return (False, "running")
