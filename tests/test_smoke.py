import asyncio
import pytest
from r2e_env.models import R2EAction
from r2e_env.environment import R2EEnv
from r2e_env.dynamics import DynamicsEngine, VALID_ACTIONS
from tasks import TASK_CONFIGS


def test_task_configs_exist():
    assert set(TASK_CONFIGS.keys()) == {"easy", "medium", "hard"}
    assert TASK_CONFIGS["easy"].max_steps == 20
    assert TASK_CONFIGS["medium"].max_steps == 30
    assert TASK_CONFIGS["hard"].max_steps == 50
    assert TASK_CONFIGS["hard"].deceptive_reward_enabled is True
    assert TASK_CONFIGS["easy"].deceptive_reward_enabled is False


def test_valid_actions_count():
    assert len(VALID_ACTIONS) == 8


def test_reset_returns_clean_observation():
    env = R2EEnv()
    obs = asyncio.run(env.reset("easy", seed=42))
    assert obs.position == 0.0
    assert obs.step_count == 0
    assert obs.failure_signal == "none"


def test_reset_determinism():
    env = R2EEnv()
    obs1 = asyncio.run(env.reset("medium", seed=42))
    obs2 = asyncio.run(env.reset("medium", seed=42))
    assert obs1 == obs2


def test_step_returns_valid_reward():
    env = R2EEnv()
    asyncio.run(env.reset("easy", seed=0))
    obs, reward, done, info = asyncio.run(env.step(R2EAction(action="probe_friction")))
    assert -1.0 <= reward <= 1.0
    assert isinstance(done, bool)


def test_probe_does_not_change_position():
    env = R2EEnv()
    asyncio.run(env.reset("hard", seed=0))
    for probe in ["probe_friction", "probe_alignment", "probe_stiffness"]:
        env2 = R2EEnv()
        obs_before = asyncio.run(env2.reset("hard", seed=0))
        obs_after, _, _, _ = asyncio.run(env2.step(R2EAction(action=probe)))
        assert obs_after.position == obs_before.position, f"{probe} changed position"


def test_absorbing_terminal_state():
    env = R2EEnv()
    asyncio.run(env.reset("easy", seed=0))
    # Force terminal via commit_solution (will fail since position < 1.0)
    obs, reward, done, info = asyncio.run(env.step(R2EAction(action="commit_solution")))
    assert done is True
    # Further steps should return done=True, reward=0.0
    obs2, reward2, done2, info2 = asyncio.run(env.step(R2EAction(action="insert")))
    assert done2 is True
    assert reward2 == 0.0


def test_invalid_action_penalty():
    env = R2EEnv()
    asyncio.run(env.reset("easy", seed=0))
    obs, reward, done, info = asyncio.run(env.step(R2EAction(action="fly_to_moon")))
    assert reward == -0.05
    assert done is False
    assert info.get("error") == "invalid_action"


def test_state_returns_dict():
    env = R2EEnv()
    asyncio.run(env.reset("hard", seed=7))
    state = env.state()
    assert "hidden_state" in state
    assert "observation" in state
    assert state["task"] == "hard"


def test_grader_end_to_end():
    import asyncio
    from r2e_env.models import R2EAction
    from r2e_env.environment import R2EEnv
    from graders import get_grader

    env = R2EEnv()
    asyncio.run(env.reset("easy", seed=42))

    # Run a short episode
    for action in ["probe_friction", "insert", "insert", "commit_solution"]:
        asyncio.run(env.step(R2EAction(action=action)))

    episode_log = env.get_episode_log()
    assert episode_log is not None

    grader = get_grader("easy")
    result = grader.grade(episode_log)

    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.success <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.correctness <= 1.0
    # Verify score formula
    expected = 0.5 * result.success + 0.3 * result.efficiency + 0.2 * result.correctness
    assert abs(result.score - expected) < 1e-9


def test_all_tasks_reset():
    import asyncio
    from r2e_env.environment import R2EEnv

    for task in ["easy", "medium", "hard"]:
        env = R2EEnv()
        obs = asyncio.run(env.reset(task, seed=0))
        assert obs.position == 0.0
        assert obs.failure_signal == "none"
        state = env.state()
        assert state["task"] == task
        assert state["hidden_state"] is not None
