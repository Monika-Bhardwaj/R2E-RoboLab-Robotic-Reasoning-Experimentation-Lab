"""
Property-based tests for R2E-RoboLab.

Property 1: Same `task` + `seed` always returns identical `Observation`
Validates: Requirements 1.1

Property 3: Probe actions do NOT change `position`
Validates: Requirements 2.3

Property 4: Total step reward ∈ [-1.0, 1.0]
Validates: Requirements 2.5

Property 7: Terminal state is absorbing
Validates: Requirements 1.4
"""
import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from r2e_env import R2EEnv, R2EAction

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


@given(seed=st.integers(0, 10000), task=st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=50)
def test_reset_determinism(seed, task):
    """
    Property 1: Same `task` + `seed` always returns identical `Observation`.
    Calling reset() twice with the same task and seed must produce equal Observations.
    Validates: Requirements 1.1
    """
    env = R2EEnv()
    obs1 = asyncio.run(env.reset(task, seed))
    obs2 = asyncio.run(env.reset(task, seed))
    assert obs1 == obs2, (
        f"task={task}, seed={seed}: reset() returned different observations: "
        f"{obs1} vs {obs2}"
    )
    env.close()


@given(seed=st.integers(0, 10000))
@settings(max_examples=50)
def test_probe_position_neutral(seed):
    """
    Property 3: Probe actions are position-neutral.
    For all three probe actions, `position` must be unchanged after the step.
    Validates: Requirements 2.3
    """
    for probe in ["probe_friction", "probe_alignment", "probe_stiffness"]:
        env = R2EEnv()
        obs_before = asyncio.run(env.reset("hard", seed=seed))
        position_before = obs_before.position
        obs_after, _, _, _ = asyncio.run(env.step(R2EAction(action=probe)))
        assert obs_after.position == position_before, (
            f"seed={seed}, probe={probe}: "
            f"position changed from {position_before} to {obs_after.position}"
        )
        env.close()


@given(actions=st.lists(st.sampled_from(VALID_ACTIONS), min_size=1, max_size=30))
@settings(max_examples=50)
def test_reward_bounds(actions):
    """
    Property 4: Total step reward ∈ [-1.0, 1.0].
    For any sequence of valid actions, each step reward must be clamped to [-1.0, 1.0].
    Validates: Requirements 2.5
    """
    env = R2EEnv()
    asyncio.run(env.reset("hard", seed=0))
    for action in actions:
        _, reward, done, _ = asyncio.run(env.step(R2EAction(action=action)))
        assert -1.0 <= reward <= 1.0, (
            f"action={action}: reward {reward} is outside [-1.0, 1.0]"
        )
        if done:
            break
    env.close()


@given(seed=st.integers(0, 10000), task=st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=50)
def test_terminal_absorbing(seed, task):
    """
    Property 7: Once done=True, further steps return done=True with reward=0.0.
    After reaching a terminal state, the environment must behave as an absorbing state.
    Validates: Requirements 1.4
    """
    env = R2EEnv()
    asyncio.run(env.reset(task, seed))

    # Drive to terminal state by calling commit_solution immediately.
    # position=0.0 so it will fail (position < 1.0), setting done=True with failure_penalty=-1.0.
    _, _, done, _ = asyncio.run(env.step(R2EAction(action="commit_solution")))
    assert done is True, (
        f"task={task}, seed={seed}: commit_solution did not set done=True"
    )

    # Verify subsequent steps are absorbing: done=True and reward=0.0
    for i in range(3):
        obs_after, reward_after, done_after, _ = asyncio.run(
            env.step(R2EAction(action="insert"))
        )
        assert done_after is True, (
            f"task={task}, seed={seed}, post-terminal step {i+1}: "
            f"expected done=True but got done={done_after}"
        )
        assert reward_after == 0.0, (
            f"task={task}, seed={seed}, post-terminal step {i+1}: "
            f"expected reward=0.0 but got reward={reward_after}"
        )

    env.close()


# ---------------------------------------------------------------------------
# Property 6: Score formula integrity
# ---------------------------------------------------------------------------

from r2e_env.models import (
    EpisodeLog,
    HiddenState,
    Observation,
    StepRecord,
)
from graders import get_grader


def _observation_strategy():
    return st.builds(
        Observation,
        position=st.floats(0.0, 1.0),
        force_feedback=st.floats(0.0, 1.0),
        lateral_instability=st.floats(0.0, 1.0),
        progress=st.floats(0.0, 1.0),
        failure_signal=st.sampled_from(["none", "jam", "slip", "unstable"]),
        last_action=st.sampled_from(
            [
                "insert",
                "adjust_left",
                "adjust_right",
                "increase_force",
                "probe_friction",
                "probe_alignment",
                "probe_stiffness",
                "commit_solution",
            ]
        ),
        step_count=st.integers(0, 50),
    )


def _step_record_strategy(step_index: int):
    return st.builds(
        StepRecord,
        step=st.just(step_index),
        action=st.sampled_from(
            [
                "insert",
                "adjust_left",
                "adjust_right",
                "increase_force",
                "probe_friction",
                "probe_alignment",
                "probe_stiffness",
                "commit_solution",
            ]
        ),
        reward=st.floats(-1.0, 1.0),
        done=st.booleans(),
        observation=_observation_strategy(),
    )


def _steps_strategy():
    return st.integers(1, 10).flatmap(
        lambda n: st.tuples(*[_step_record_strategy(i) for i in range(n)]).map(list)
    )


def _hidden_state_strategy():
    return st.builds(
        HiddenState,
        friction=st.sampled_from(["low", "high"]),
        alignment=st.sampled_from(["aligned", "misaligned"]),
        stiffness=st.sampled_from(["rigid", "compliant"]),
    )


def _episode_log_strategy():
    return st.builds(
        EpisodeLog,
        task=st.sampled_from(["easy", "medium", "hard"]),
        seed=st.integers(0, 10000),
        steps=_steps_strategy(),
        final_obs=_observation_strategy(),
        total_reward=st.floats(-50.0, 50.0),
        done=st.booleans(),
        hidden_state=_hidden_state_strategy(),
    )


@given(episode=_episode_log_strategy())
@settings(max_examples=100)
def test_score_formula_integrity(episode):
    """
    Property 6: score == 0.5*success + 0.3*efficiency + 0.2*correctness within 1e-9.
    Validates: Requirements 4.1
    """
    grader = get_grader(episode.task)
    result = grader.grade(episode)

    expected = 0.5 * result.success + 0.3 * result.efficiency + 0.2 * result.correctness
    # The grader clamps to [0, 1], so expected may differ from raw formula only at boundaries.
    # We verify the formula holds before clamping by checking the clamped result matches.
    clamped_expected = min(1.0, max(0.0, expected))
    assert abs(result.score - clamped_expected) < 1e-9, (
        f"score={result.score} != clamp(0.5*{result.success} + 0.3*{result.efficiency} "
        f"+ 0.2*{result.correctness}) = {clamped_expected}"
    )
