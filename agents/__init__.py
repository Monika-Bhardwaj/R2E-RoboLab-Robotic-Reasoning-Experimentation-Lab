"""
Five agent baselines for R2E-RoboLab v2 evaluation.

1. RandomAgent        — random actions, no reasoning
2. GreedyAgent        — always increase_force (falls into deceptive trap)
3. DeterministicAgent — optimal probe-then-act, no reasoning text
4. ReasoningAgent     — deterministic + structured <think> reasoning (no LLM)
5. OracleAgent        — has access to hidden state (theoretical upper bound)
"""
import random
from r2e_env.models import R2EAction, Observation, HiddenState

VALID_ACTIONS = [
    "insert", "adjust_left", "adjust_right", "increase_force",
    "probe_friction", "probe_alignment", "probe_stiffness", "commit_solution",
]


class RandomAgent:
    """Baseline: completely random actions, no reasoning."""
    name = "random"

    def act(self, obs: Observation, probed: set, **kwargs) -> R2EAction:
        return R2EAction(action=random.choice(VALID_ACTIONS), reasoning="")


class GreedyAgent:
    """Baseline: always uses increase_force to maximize immediate reward.
    Falls into the deceptive trap on hard tasks."""
    name = "greedy"

    def act(self, obs: Observation, probed: set, **kwargs) -> R2EAction:
        if obs.position >= 1.0 and obs.failure_signal == "none":
            return R2EAction(action="commit_solution", reasoning="")
        return R2EAction(action="increase_force", reasoning="")


class DeterministicAgent:
    """Optimal probe-then-act strategy. No reasoning text.
    Represents the ceiling for a non-LLM rule-based agent."""
    name = "deterministic"

    def act(self, obs: Observation, probed: set, **kwargs) -> R2EAction:
        # Phase 1: Probe all unknowns
        if "friction" not in probed:
            probed.add("friction")
            return R2EAction(action="probe_friction", reasoning="")
        if "alignment" not in probed:
            probed.add("alignment")
            return R2EAction(action="probe_alignment", reasoning="")
        if "stiffness" not in probed:
            probed.add("stiffness")
            return R2EAction(action="probe_stiffness", reasoning="")

        # Fix misalignment
        if obs.lateral_instability > 0.3:
            return R2EAction(action="adjust_left", reasoning="")

        # Commit when done
        if obs.position >= 1.0 and obs.failure_signal == "none":
            return R2EAction(action="commit_solution", reasoning="")

        # Insert safely
        high_friction = obs.force_feedback > 0.5
        compliant = obs.known_variables.get("stiffness") == "compliant"
        if high_friction and compliant:
            return R2EAction(action="insert", reasoning="")
        return R2EAction(action="increase_force", reasoning="")


class ReasoningAgent:
    """Optimal probe-then-act with structured <think> reasoning.
    This is the 'trained model' target behavior — shows what we want the LLM to learn."""
    name = "reasoning"

    def act(self, obs: Observation, probed: set, **kwargs) -> R2EAction:
        # Build reasoning trace
        lines = []
        lines.append(f"Step {obs.step_count}: position={obs.position:.2f}, "
                     f"force_feedback={obs.force_feedback:.2f}, "
                     f"lateral_instability={obs.lateral_instability:.2f}, "
                     f"failure={obs.failure_signal}")

        known = obs.known_variables

        if "friction" not in probed:
            lines.append("I have not probed friction yet. Probing is essential before applying force.")
            lines.append("Action plan: probe_friction")
            probed.add("friction")
            return R2EAction(action="probe_friction", reasoning="\n".join(lines))

        if "alignment" not in probed:
            lines.append(f"Friction known={known.get('friction','?')}. Must check alignment next.")
            probed.add("alignment")
            return R2EAction(action="probe_alignment", reasoning="\n".join(lines))

        if "stiffness" not in probed:
            lines.append(f"Friction={known.get('friction','?')}, alignment={known.get('alignment','?')}. "
                         "Checking stiffness before applying force.")
            probed.add("stiffness")
            return R2EAction(action="probe_stiffness", reasoning="\n".join(lines))

        high_friction = known.get("friction") == "high"
        compliant = known.get("stiffness") == "compliant"
        misaligned = obs.lateral_instability > 0.3

        if misaligned:
            lines.append(f"lateral_instability={obs.lateral_instability:.2f} > 0.3, indicates misalignment. Adjusting.")
            return R2EAction(action="adjust_left", reasoning="\n".join(lines))

        if obs.position >= 1.0 and obs.failure_signal == "none":
            lines.append("Position=1.0 and no failure. Safe to commit.")
            return R2EAction(action="commit_solution", reasoning="\n".join(lines))

        if high_friction and compliant:
            lines.append("HIGH FRICTION + COMPLIANT STIFFNESS detected. increase_force is DANGEROUS — would cause jam.")
            lines.append("Using safe insert instead.")
            return R2EAction(action="insert", reasoning="\n".join(lines))

        lines.append("Physical conditions are safe. Using increase_force for efficiency.")
        return R2EAction(action="increase_force", reasoning="\n".join(lines))


class OracleAgent:
    """Has access to hidden state — theoretical upper bound.
    Uses perfect knowledge to make optimal decisions every time."""
    name = "oracle"

    def act(self, obs: Observation, probed: set, hidden: HiddenState = None, **kwargs) -> R2EAction:
        if hidden is None:
            return R2EAction(action="insert", reasoning="")

        # Immediately knows everything — skip probing
        if obs.lateral_instability > 0.3 and hidden.alignment_error.value == "misaligned":
            return R2EAction(action="adjust_left", reasoning="oracle: misaligned, adjusting")

        if obs.position >= 1.0 and obs.failure_signal == "none":
            return R2EAction(action="commit_solution", reasoning="oracle: complete")

        dangerous = (
            hidden.friction_level.value == "high"
            and hidden.stiffness.value == "compliant"
        )
        if dangerous:
            return R2EAction(action="insert", reasoning="oracle: dangerous conditions, safe insert")
        return R2EAction(action="increase_force", reasoning="oracle: safe to force")
