import statistics
from r2e_env.models import EpisodeLog, GradeResult
from tasks import TASK_CONFIGS


class BaseGrader:
    required_probes: set[str] = set()

    def grade(self, episode: EpisodeLog) -> GradeResult:
        task_config = TASK_CONFIGS[episode.task]

        # --- Success sub-score (40%) ---
        success = 1.0 if (
            episode.done
            and episode.final_obs.failure_signal == "none"
            and episode.final_obs.position >= 1.0
        ) else 0.0

        # --- Efficiency sub-score (25%) ---
        max_steps = task_config.max_steps
        actual_steps = len(episode.steps)
        efficiency = max(0.0, 1.0 - (actual_steps / max_steps))

        # --- Correctness sub-score (20%) ---
        probe_actions_used = [
            s.action for s in episode.steps
            if s.action.startswith("probe_")
        ]
        unique_probes = set(a.replace("probe_", "") for a in probe_actions_used)
        unnecessary_probes = max(0, len(probe_actions_used) - len(self.required_probes))

        trap_falls = sum(
            1 for s in episode.steps
            if s.action == "increase_force"
            and s.done is True
            and s.observation.failure_signal == "jam"
        )

        correctness = 1.0
        correctness -= 0.1 * unnecessary_probes
        correctness -= 0.3 * trap_falls
        correctness = max(0.0, correctness)

        # --- Reasoning sub-score (15%) — Phase 2 ---
        reasoning_score = 0.0
        if episode.reasoning_scores:
            # Mean reasoning quality, normalized to [0, 1]
            raw = statistics.mean(episode.reasoning_scores)
            # raw is in [-0.20, +0.20], normalize to [0, 1]
            reasoning_score = max(0.0, min(1.0, (raw + 0.20) / 0.40))

        # --- Composite score ---
        # Weights: success=0.40, efficiency=0.25, correctness=0.20, reasoning=0.15
        score = (
            0.40 * success
            + 0.25 * efficiency
            + 0.20 * correctness
            + 0.15 * reasoning_score
        )
        score = min(1.0, max(0.0, score))

        return GradeResult(
            score=score,
            success=success,
            efficiency=efficiency,
            correctness=correctness,
            reasoning_score=reasoning_score,
            details={
                "actual_steps": actual_steps,
                "max_steps": max_steps,
                "probe_actions_used": probe_actions_used,
                "unique_probes": list(unique_probes),
                "required_probes": list(self.required_probes),
                "unnecessary_probes": unnecessary_probes,
                "trap_falls": trap_falls,
                "mean_reasoning_quality": statistics.mean(episode.reasoning_scores) if episode.reasoning_scores else 0.0,
            }
        )
