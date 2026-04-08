from r2e_env.models import EpisodeLog, GradeResult
from tasks import TASK_CONFIGS


class BaseGrader:
    required_probes: set[str] = set()

    def grade(self, episode: EpisodeLog) -> GradeResult:
        task_config = TASK_CONFIGS[episode.task]

        # Success sub-score
        success = 1.0 if (
            episode.done
            and episode.final_obs.failure_signal == "none"
            and episode.final_obs.position >= 1.0
        ) else 0.0

        # Efficiency sub-score (fewer steps = higher score)
        max_steps = task_config.max_steps
        actual_steps = len(episode.steps)
        efficiency = max(0.0, 1.0 - (actual_steps / max_steps))

        # Correctness sub-score
        probe_actions_used = [
            s.action for s in episode.steps
            if s.action.startswith("probe_")
        ]

        # Count unnecessary probes (probes beyond the required set)
        unnecessary_probes = max(0, len(probe_actions_used) - len(self.required_probes))

        # Count deceptive trap falls (increase_force that resulted in jam)
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

        score = 0.5 * success + 0.3 * efficiency + 0.2 * correctness
        score = min(1.0, max(0.0, score))

        return GradeResult(
            score=score,
            success=success,
            efficiency=efficiency,
            correctness=correctness,
            details={
                "actual_steps": actual_steps,
                "max_steps": max_steps,
                "probe_actions_used": probe_actions_used,
                "required_probes": list(self.required_probes),
                "unnecessary_probes": unnecessary_probes,
                "trap_falls": trap_falls,
            }
        )
