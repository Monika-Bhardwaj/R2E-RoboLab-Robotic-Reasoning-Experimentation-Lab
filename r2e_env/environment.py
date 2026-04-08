from r2e_env.models import Observation, R2EAction, HiddenState, StepRecord, EpisodeLog, TaskConfig
from r2e_env.dynamics import DynamicsEngine, VALID_ACTIONS
from tasks import TASK_CONFIGS


class R2EEnv:
    def __init__(self):
        self._dynamics = DynamicsEngine()
        self._obs: Observation | None = None
        self._hidden: HiddenState | None = None
        self._task_config: TaskConfig | None = None
        self._task: str | None = None
        self._seed: int | None = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._revealed_probes: set[str] = set()
        self._done: bool = False
        self._steps: list[StepRecord] = []

    async def reset(self, task: str, seed: int = 42) -> Observation:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task!r}. Must be one of {list(TASK_CONFIGS)}")

        self._task = task
        self._seed = seed
        self._task_config = TASK_CONFIGS[task]
        self._step_count = 0
        self._total_reward = 0.0
        self._revealed_probes = set()
        self._done = False
        self._steps = []

        self._hidden = self._dynamics.sample_hidden_state(seed, self._task_config)

        self._obs = Observation(
            position=0.0,
            force_feedback=0.5,
            lateral_instability=0.0,
            progress=0.0,
            failure_signal="none",
            last_action="none",
            step_count=0,
        )
        return self._obs

    async def step(self, action: R2EAction) -> tuple[Observation, float, bool, dict]:
        # Absorbing terminal state
        if self._done:
            return (self._obs, 0.0, True, {"reason": "already_done"})

        if self._hidden is None:
            raise RuntimeError("Must call reset() before step()")

        # Invalid action
        if action.action not in VALID_ACTIONS:
            reward = -0.05
            info = {"error": "invalid_action"}
            record = StepRecord(step=self._step_count + 1, action=action.action, reward=reward, done=False, observation=self._obs)
            self._steps.append(record)
            return (self._obs, reward, False, info)

        next_obs, rc = self._dynamics.apply_action(
            action.action, self._hidden, self._obs, self._revealed_probes, self._task_config
        )

        self._step_count += 1
        next_obs.step_count = self._step_count

        reward = max(-1.0, min(1.0, rc.total()))
        self._total_reward += reward
        self._obs = next_obs

        # Check terminal conditions
        done, reason = self._dynamics.is_terminal(next_obs)

        # commit_solution success/failure also terminates
        if action.action == "commit_solution":
            done = True
            reason = "commit_solution"

        # Max steps
        if self._step_count >= self._task_config.max_steps:
            done = True
            reason = "max_steps"

        self._done = done

        record = StepRecord(step=self._step_count, action=action.action, reward=reward, done=done, observation=next_obs)
        self._steps.append(record)

        return (next_obs, reward, done, {"reason": reason})

    def state(self) -> dict:
        return {
            "task": self._task,
            "seed": self._seed,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "done": self._done,
            "revealed_probes": list(self._revealed_probes),
            "hidden_state": self._hidden.model_dump() if self._hidden else None,
            "observation": self._obs.model_dump() if self._obs else None,
        }

    async def close(self) -> None:
        pass  # no-op resource release

    def get_episode_log(self) -> EpisodeLog | None:
        if self._hidden is None or self._obs is None:
            return None
        return EpisodeLog(
            task=self._task,
            seed=self._seed,
            steps=self._steps,
            final_obs=self._obs,
            total_reward=self._total_reward,
            done=self._done,
            revealed_probes=list(self._revealed_probes),
            hidden_state=self._hidden,
        )
