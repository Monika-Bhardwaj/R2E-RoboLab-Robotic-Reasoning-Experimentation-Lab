"""
inference.py — R2E-RoboLab v2 baseline inference script

Runs all 3 tasks (easy, medium, hard) sequentially using an LLM agent
with MANDATORY structured reasoning (<think>...</think> before every action).
Emits structured stdout logs in the required [START]/[STEP]/[END] format.

Environment variables:
  API_BASE_URL  LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      Hugging Face / API key (also checked as API_KEY)
"""

import asyncio
import os
import re
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

TASK_MAX_STEPS = {"easy": 40, "medium": 60, "hard": 90}
TASK_SEEDS = {"easy": 42, "medium": 42, "hard": 42}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent controlling a robotic arm performing a precision insertion task.
    Goal: Fully insert the component (position=1.0) and call 'commit_solution'.

    HIDDEN physical properties you must discover through probing:
    - friction_level: affects insertion speed. Reveal with 'probe_friction'.
      force_feedback > 0.7 means HIGH friction; < 0.3 means LOW friction.
    - alignment_error: affects stability. Reveal with 'probe_alignment'.
      lateral_instability > 0.5 means MISALIGNED; < 0.2 means ALIGNED.
    - stiffness: affects jam risk. Reveal with 'probe_stiffness'.
      A spike in lateral_instability after probing means COMPLIANT stiffness.

    Available actions:
    - insert              Progress +0.20 (low friction) or +0.10 (high friction). Fails if unstable.
    - adjust_left         Correct leftward misalignment.
    - adjust_right        Correct rightward misalignment.
    - increase_force      FAST progress (+0.40) but JAMS if friction=HIGH and stiffness=COMPLIANT.
    - probe_friction      Reveals friction via force_feedback reading.
    - probe_alignment     Reveals alignment via lateral_instability reading.
    - probe_stiffness     Reveals stiffness via instability spike pattern.
    - retract             Recovers from a 'wedged' failure state.
    - commit_solution     Terminate episode (only safe when position=1.0 and failure_signal=none).

    MANDATORY REASONING PROTOCOL:
    You MUST think step by step before every action. Output format:
    <think>
    [Your reasoning about the current physical state and what action is safe/optimal]
    </think>
    Action: [exactly one action name]

    STRATEGY:
    1. Always probe friction, alignment, and stiffness in the first 3 steps.
    2. If lateral_instability > 0.3, use adjust_left or adjust_right.
    3. NEVER use increase_force if friction=high AND stiffness=compliant (causes JAM).
    4. Use increase_force only when physical conditions are confirmed safe.
    5. Once position >= 1.0 and failure_signal=none, call commit_solution.
""").strip()


# ── Logging helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ──────────────────────────────────────────────────────────────
def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    known = obs_dict.get("known_variables", {})
    known_str = ", ".join(f"{k}={v}" for k, v in known.items()) if known else "none yet"
    phase = obs_dict.get("phase", "investigation")
    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""
        Step {step} | Phase: {phase}

        Current observation:
          position:            {obs_dict['position']:.3f}
          force_feedback:      {obs_dict['force_feedback']:.3f}
          lateral_instability: {obs_dict['lateral_instability']:.3f}
          failure_signal:      {obs_dict['failure_signal']}
          last_action:         {obs_dict['last_action']}

        Known physical properties: {known_str}
        Last reward: {last_reward:.3f}

        Recent history:
        {history_block}

        Think carefully, then choose your action:
    """).strip()


# ── LLM response parser ─────────────────────────────────────────────────────────
def parse_llm_response(raw: str) -> tuple[str, str]:
    """Extract <think> reasoning and Action: from raw LLM response."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    reasoning = think_match.group(1).strip() if think_match else ""

    action_match = re.search(r"\bAction:\s*([a-z_]+)", raw, re.IGNORECASE)
    if action_match:
        return reasoning, action_match.group(1).strip().lower()

    # Fallback: find any valid action name in the response
    valid = [
        "commit_solution", "increase_force", "probe_friction",
        "probe_alignment", "probe_stiffness", "adjust_left",
        "adjust_right", "insert",
    ]
    for action in valid:
        if action in raw.lower():
            return reasoning, action

    return reasoning, "probe_friction"


# ── Deterministic fallback (used when API fails) ────────────────────────────────
def build_fallback_reasoning(obs_dict: dict, probed: set) -> str:
    """Build a reasoning trace for the deterministic fallback action."""
    lines = []
    pos = obs_dict["position"]
    force = obs_dict["force_feedback"]
    instability = obs_dict["lateral_instability"]
    failure = obs_dict["failure_signal"]
    known = obs_dict.get("known_variables", {})

    lines.append(f"[Fallback reasoning] position={pos:.2f}, force={force:.2f}, instability={instability:.2f}")

    if "friction" not in probed:
        lines.append("Friction not probed yet. Must probe before applying force.")
    elif "alignment" not in probed:
        lines.append(f"Friction known={known.get('friction','?')}. Alignment not probed yet.")
    elif "stiffness" not in probed:
        lines.append(f"Friction={known.get('friction','?')}, alignment={known.get('alignment','?')}. Checking stiffness.")
    elif instability > 0.3:
        lines.append(f"lateral_instability={instability:.2f} > 0.3 indicates misalignment. Adjusting.")
    elif pos >= 1.0 and failure == "none":
        lines.append("Position=1.0 and no failure. Committing solution.")
    else:
        high_friction = force > 0.5
        compliant = known.get("stiffness") == "compliant"
        if high_friction and compliant:
            lines.append("HIGH FRICTION + COMPLIANT detected. increase_force is DANGEROUS. Using safe insert.")
        else:
            lines.append("Physical conditions safe. Using increase_force for efficiency.")

    return "\n".join(lines)


_probed_vars: set = set()


def deterministic_fallback(obs_dict: dict, probed: set) -> tuple[str, str]:
    """Returns (reasoning, action)."""
    position = obs_dict["position"]
    force = obs_dict["force_feedback"]
    instability = obs_dict["lateral_instability"]
    failure = obs_dict["failure_signal"]
    known = obs_dict.get("known_variables", {})

    reasoning = build_fallback_reasoning(obs_dict, probed)

    if "friction" not in probed:
        probed.add("friction")
        return reasoning, "probe_friction"
    if "alignment" not in probed:
        probed.add("alignment")
        return reasoning, "probe_alignment"
    if "stiffness" not in probed:
        probed.add("stiffness")
        return reasoning, "probe_stiffness"
    if instability > 0.3:
        return reasoning, "adjust_left"
    if position >= 1.0 and failure == "none":
        return reasoning, "commit_solution"

    high_friction = force > 0.5
    compliant = known.get("stiffness") == "compliant"
    if high_friction and compliant:
        return reasoning, "insert"
    return reasoning, "increase_force"


# ── LLM action getter ───────────────────────────────────────────────────────────
def get_model_action(
    client: OpenAI,
    step: int,
    obs_dict: dict,
    last_reward: float,
    history: List[str],
) -> tuple[str, str]:
    """Returns (reasoning, action). Falls back to deterministic if API fails."""
    global _probed_vars
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
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
                temperature=0.7,
                max_tokens=300,   # enough for <think> + Action:
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            reasoning, action = parse_llm_response(raw)

            if action.startswith("probe_"):
                _probed_vars.add(action.replace("probe_", ""))

            return reasoning, action

        except Exception as exc:
            err = str(exc).lower()
            if "429" in str(exc) or "rate" in err or "exhaust" in err or "limit" in err:
                delay = base_delay * (2 ** attempt)
                print(f"[DEBUG] Rate limited. Retry in {delay:.0f}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue

            print(f"[DEBUG] API failed (attempt {attempt+1}): {exc}", flush=True)
            return deterministic_fallback(obs_dict, _probed_vars)

    print("[DEBUG] All retries exhausted, using deterministic fallback", flush=True)
    return deterministic_fallback(obs_dict, _probed_vars)


# ── Task runner ─────────────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task: str, seed: int) -> None:
    global _probed_vars
    _probed_vars = set()

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

            reasoning, action_str = get_model_action(client, step, obs_dict, last_reward, history)
            action = R2EAction(action=action_str, reasoning=reasoning)

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
                f"Step {step}: {action_str} → reward {reward:+.3f}, "
                f"pos={obs_dict['position']:.2f}, fail={obs_dict['failure_signal']}, "
                f"known={obs_dict.get('known_variables', {})}"
            )

        # Grade the episode
        episode_log = env.get_episode_log()
        if episode_log:
            grade_result = grader.grade(episode_log)
            score = grade_result.score
            success = grade_result.success == 1.0
            # Print sub-scores for transparency
            print(
                f"[GRADE] task={task} success={grade_result.success:.2f} "
                f"efficiency={grade_result.efficiency:.2f} "
                f"correctness={grade_result.correctness:.2f} "
                f"reasoning={grade_result.reasoning_score:.2f}",
                flush=True,
            )

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ─────────────────────────────────────────────────────────────────
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
