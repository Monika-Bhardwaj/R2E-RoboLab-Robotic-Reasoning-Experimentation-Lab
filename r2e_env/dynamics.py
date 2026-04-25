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


def score_reasoning(
    reasoning: str,
    obs: Observation,
    hidden: HiddenState,
    action: str,
) -> float:
    """
    Score the quality of the agent's <think> reasoning relative to ground truth.
    Returns a value in [-0.20, +0.20].

    This is the key Phase 2 innovation: reward correct causal inference,
    penalize hallucination and dangerous overconfidence.
    """
    if not reasoning:
        return 0.0

    text = reasoning.lower()
    score = 0.0

    # --- Positive signals ---

    # R1: High instability visible → agent mentions it and plans to probe/adjust
    if obs.lateral_instability > 0.5:
        if any(w in text for w in ["instability", "lateral", "unstable", "misalign"]):
            score += 0.04

    # R2: Agent correctly warns increase_force is dangerous (when it IS dangerous)
    is_dangerous = (
        hidden.friction_level == Friction.high
        and hidden.stiffness == Stiffness.compliant
    )
    if is_dangerous and "increase_force" in text:
        if any(w in text for w in ["danger", "risky", "avoid", "careful", "jam", "unsafe", "not safe"]):
            score += 0.08

    # R3: Early steps → agent plans to probe before acting
    if obs.step_count < 4:
        if "probe" in text:
            score += 0.04

    # R4: Agent correctly identifies friction from probe reading
    if obs.force_feedback > 0.7 and hidden.friction_level == Friction.high:
        if any(w in text for w in ["high friction", "friction is high", "friction: high"]):
            score += 0.05
    if obs.force_feedback < 0.3 and hidden.friction_level == Friction.low:
        if any(w in text for w in ["low friction", "friction is low", "friction: low"]):
            score += 0.05

    # R5: Agent correctly identifies misalignment from lateral instability
    if obs.lateral_instability > 0.6 and hidden.alignment_error == Alignment.misaligned:
        if any(w in text for w in ["misalign", "alignment", "adjust", "lateral"]):
            score += 0.04

    # R6: Agent plans commit when position is near 1.0
    if obs.position >= 0.9 and obs.failure_signal == "none":
        if "commit" in text:
            score += 0.03

    # --- Negative signals (hallucination / contradiction) ---

    # P1: Claims friction is LOW when it's HIGH (and we've already probed)
    if (hidden.friction_level == Friction.high
            and obs.force_feedback > 0.7   # probe already done
            and any(w in text for w in ["friction is low", "low friction", "friction: low"])):
        score -= 0.10

    # P2: Plans increase_force without any safety check when dangerous
    if is_dangerous and action == "increase_force":
        if not any(w in text for w in ["probe", "check", "safe", "verify", "confirm"]):
            score -= 0.10

    # P3: Claims position is high when it's low (confabulation)
    if obs.position < 0.2 and any(w in text for w in ["almost done", "nearly complete", "position is high"]):
        score -= 0.05

    return max(-0.20, min(0.20, score))


class DynamicsEngine:
    def sample_hidden_state(self, seed: int, task_config: TaskConfig) -> HiddenState:
        import hashlib

        def rng_for(variable: str) -> random.Random:
            # SHA-256 ensures truly independent seeds per variable
            h = hashlib.sha256(f"{seed}:{variable}".encode()).hexdigest()
            return random.Random(int(h[:16], 16))

        friction = Friction.low
        alignment = Alignment.aligned
        stiffness = Stiffness.rigid

        if "friction_level" in task_config.active_variables:
            friction = rng_for("friction").choice([Friction.low, Friction.high])
        if "alignment_error" in task_config.active_variables:
            alignment = rng_for("alignment").choice([Alignment.aligned, Alignment.misaligned])
        if "stiffness" in task_config.active_variables:
            stiffness = rng_for("stiffness").choice([Stiffness.rigid, Stiffness.compliant])

        return HiddenState(
            friction_level=friction,
            alignment_error=alignment,
            stiffness=stiffness,
        )


    def get_phase(self, step_count: int, task_config: TaskConfig) -> str:
        inv_boundary = int(task_config.max_steps * task_config.investigation_fraction)
        ver_boundary = int(task_config.max_steps * (
            task_config.investigation_fraction + task_config.verification_fraction
        ))
        if step_count < inv_boundary:
            return "investigation"
        elif step_count < ver_boundary:
            return "verification"
        else:
            return "execution"

    def apply_action(
        self,
        action: str,
        reasoning: str,
        hidden: HiddenState,
        obs: Observation,
        revealed_probes: set[str],
        known_variables: dict[str, str],
        task_config: TaskConfig,
    ) -> tuple[Observation, RewardComponents]:
        rc = RewardComponents()
        next_obs = obs.model_copy(deep=True)
        rc.step_penalty = -0.01

        # Score reasoning quality
        rc.reasoning_quality = score_reasoning(reasoning, obs, hidden, action)

        if action == "insert":
            if obs.is_wedged:
                rc.invalid_penalty = -0.1
            elif hidden.alignment_error == Alignment.misaligned and next_obs.lateral_instability > 0.3:
                next_obs.is_wedged = True
                next_obs.failure_signal = "wedged"
                rc.invalid_penalty = -0.05
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
                next_obs.lateral_instability = min(1.0, obs.lateral_instability + 0.1)

        elif action == "increase_force":
            if obs.is_wedged:
                rc.invalid_penalty = -0.1
            else:
                delta = 0.40
                next_obs.position = min(1.0, obs.position + delta)
                rc.progress_reward = 0.1 * delta  # +0.04

            if (
                hidden.friction_level == Friction.high
                and hidden.stiffness == Stiffness.compliant
                and task_config.deceptive_reward_enabled
            ):
                next_obs.failure_signal = "jam"
                rc.failure_penalty = -1.0

        elif action == "retract":
            if obs.is_wedged:
                next_obs.is_wedged = False
                next_obs.failure_signal = "none"
            else:
                rc.invalid_penalty = -0.05

        elif action == "probe_friction":
            next_obs.force_feedback = 0.9 if hidden.friction_level == Friction.high else 0.2
            if "friction" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("friction")
            known_variables["friction"] = hidden.friction_level.value

        elif action == "probe_alignment":
            next_obs.lateral_instability = (
                0.8 if hidden.alignment_error == Alignment.misaligned else 0.1
            )
            if "alignment" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("alignment")
            known_variables["alignment"] = hidden.alignment_error.value

        elif action == "probe_stiffness":
            if hidden.stiffness == Stiffness.compliant:
                next_obs.lateral_instability = min(1.0, obs.lateral_instability + 0.4)
            else:
                next_obs.lateral_instability = max(0.0, obs.lateral_instability - 0.1)
            if "stiffness" not in revealed_probes:
                rc.probe_reward = 0.05
                revealed_probes.add("stiffness")
            known_variables["stiffness"] = hidden.stiffness.value

        elif action == "commit_solution":
            if next_obs.position >= 1.0 and next_obs.failure_signal == "none":
                rc.success_reward = 1.0
            else:
                rc.failure_penalty = -1.0

        next_obs.last_action = action
        next_obs.step_count = obs.step_count + 1
        next_obs.progress = next_obs.position
        next_obs.known_variables = dict(known_variables)
        next_obs.last_reasoning = reasoning[:200] if reasoning else ""
        next_obs.phase = self.get_phase(next_obs.step_count, task_config)

        if task_config.sparse_reward_mode and action != "commit_solution":
            # Zero out all intermediate rewards for Super Long-Horizon sparse reward mode
            rc.progress_reward = 0.0
            rc.step_penalty = 0.0
            rc.invalid_penalty = 0.0
            rc.probe_reward = 0.0
            # Keep reasoning_quality as an intrinsic signal if desired, or zero it
            # We'll leave reasoning_quality so GRPO still aligns thought formatting
            
        return (next_obs, rc)

    def is_terminal(self, obs: Observation) -> tuple[bool, str]:
        if obs.failure_signal == "jam":
            return (True, "jam")
        if obs.failure_signal == "slip":
            return (True, "slip")
        return (False, "running")
