"""
inference.py — R2E baseline inference script

Runs all 3 tasks (easy, medium, hard) sequentially using an LLM agent
and emits structured stdout logs in the required format.

Environment variables:
  API_BASE_URL  LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      Hugging Face / API key (also checked as API_KEY)
"""

import asyncio
import os
import textwrap
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from r2e_env import R2EAction, R2EEnv
from graders import get_grader

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "r2e_env"

TASK_MAX_STEPS = {"easy": 30, "medium": 40, "hard": 60}
TASK_SEEDS = {"easy": 42, "medium": 42, "hard": 42}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent controlling a robotic arm performing a precision insertion task.
    Goal: Fully insert the component (position=1.0) and call 'commit_solution'.

    Hidden physical properties you must discover:
    - friction: affects insertion speed. Probe with 'probe_friction'.
    - alignment: affects stability. Probe with 'probe_alignment'.
    - stiffness: affects jam risk. Probe with 'probe_stiffness'.

    Actions:
    - insert: Progress +0.20 (low friction), +0.10 (high friction). Fails if unstable.
    - adjust_left / adjust_right: Corrects misalignment (lateral_instability).
    - increase_force: FAST progress (+0.40) but JAMS if friction=high and stiffness=compliant.
    - probe_friction: force_feedback: 0.9=high, 0.2=low.
    - probe_alignment: lateral_instability: 0.8=misaligned, 0.1=aligned.
    - probe_stiffness: instability spike if compliant.
    - commit_solution: terminate (only when position=1.0).

    CRITICAL STRATEGY:
    1. ALWAYS probe friction, alignment, and stiffness at the start.
    2. If lateral_instability > 0.3, use adjust_left or adjust_right until it is 0.1.
    3. If friction is low OR stiffness is rigid, use 'increase_force' for speed.
    4. If friction is high AND stiffness is compliant, NEVER use 'increase_force'. Use 'insert'.
    5. Once position is 1.0, call 'commit_solution'.

    Respond with EXACTLY one action string. No explanation.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # MANDATORY: Double space after [STEP]
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # MANDATORY: .2f for score and rewards
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Current observation:
          position: {obs_dict['position']:.3f}
          force_feedback: {obs_dict['force_feedback']:.3f}
          lateral_instability: {obs_dict['lateral_instability']:.3f}
          failure_signal: {obs_dict['failure_signal']}
          last_action: {obs_dict['last_action']}
          step_count: {obs_dict['step_count']}
        Last reward: {last_reward:.2f}

        Recent history:
        {history_block}

        Choose your next action:
    """).strip()


def deterministic_fallback(obs_dict: dict, probed: set) -> str:
    """
    Smart fallback strategy when the LLM API is unavailable.
    Follows the optimal Probe → Infer → Act sequence to maintain high scores.
    """
    position = obs_dict["position"]
    force = obs_dict["force_feedback"]
    instability = obs_dict["lateral_instability"]
    failure = obs_dict["failure_signal"]

    # Phase 1: Probe all unknowns first (costs 3 steps, earns +0.05 each)
    if "friction" not in probed:
        probed.add("friction")
        return "probe_friction"
    if "alignment" not in probed:
        probed.add("alignment")
        return "probe_alignment"
    if "stiffness" not in probed:
        probed.add("stiffness")
        return "probe_stiffness"

    # Phase 2: Fix misalignment if detected
    if instability > 0.3:
        return "adjust_left"

    # Phase 3: Commit if we're at the goal
    if position >= 1.0 and failure == "none":
        return "commit_solution"

    # Phase 4: Choose insertion strategy based on probed data
    high_friction = force > 0.5
    compliant = instability > 0.2  # after stiffness probe, compliant shows a spike

    if high_friction and compliant:
        # DANGEROUS combo — never use increase_force, use safe insert
        return "insert"
    else:
        # Safe to use force for speed
        return "increase_force"


# Track probed variables across steps (reset per task in run_task)
_probed_vars: set = set()


def get_model_action(client: OpenAI, step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    global _probed_vars
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    valid_actions = [
        "insert", "adjust_left", "adjust_right", "increase_force",
        "probe_friction", "probe_alignment", "probe_stiffness", "commit_solution"
    ]

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=20,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip().lower()
            for action in valid_actions:
                if action in text:
                    # Track probes even when LLM is working
                    if action.startswith("probe_"):
                        _probed_vars.add(action.replace("probe_", ""))
                    return action
            return deterministic_fallback(obs_dict, _probed_vars)
        except Exception as exc:
            err = str(exc).lower()
            if "429" in str(exc) or "rate" in err or "exhaust" in err or "limit" in err:
                delay = base_delay * (2 ** attempt)
                print(f"[DEBUG] Rate limited. Retry in {delay:.0f}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue

            print(f"[DEBUG] API failed, using deterministic fallback: {exc}", flush=True)
            return deterministic_fallback(obs_dict, _probed_vars)

    # All retries exhausted — use smart fallback
    print("[DEBUG] All retries exhausted, using deterministic fallback", flush=True)
    return deterministic_fallback(obs_dict, _probed_vars)


async def run_task(client: OpenAI, task: str, seed: int) -> None:
    global _probed_vars
    _probed_vars = set()  # Reset for each new task
    env = R2EEnv()
    grader = get_grader(task)
    max_steps = TASK_MAX_STEPS[task]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset(task=task, seed=seed)
        obs_dict = obs.model_dump()
        last_reward = 0.0
        history: List[str] = []
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action_str = get_model_action(client, step, obs_dict, last_reward, history)
            action = R2EAction(action=action_str)

            obs, reward, done, info = await env.step(action)
            obs_dict = obs.model_dump()
            error = info.get("error", None)
            if error is None and obs_dict["failure_signal"] != "none":
                error = obs_dict["failure_signal"]

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f}, "
                f"pos={obs_dict['position']:.2f}, fail={obs_dict['failure_signal']}"
            )

        # Grade the episode
        episode_log = env.get_episode_log()
        if episode_log:
            grade_result = grader.grade(episode_log)
            score = grade_result.score
            success = grade_result.success == 1.0

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ["easy", "medium", "hard"]:
        seed = TASK_SEEDS[task]
        try:
            await run_task(client, task, seed)
        except Exception as e:
            print(f"[ERROR] Task {task} failed with: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
