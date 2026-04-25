"""
build_notebook.py — generates train_r2e_grpo.ipynb
Run: python build_notebook.py
"""
import json

def cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }

def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


# ─────────────────────────────────────────────────────────────────────────────
CELL1 = """\
# Cell 1: Install dependencies (pinned for stability)
!pip install -q modelscope
!pip install -q unsloth
!pip install -q "trl>=0.11.0" transformers accelerate peft datasets matplotlib
print("All packages installed.")
"""

CELL2 = """\
# Cell 2: Load HuggingFace token from Colab Secrets
# Add your token: click the 🔑 icon in the left sidebar -> + Add new secret
#   Name: HF_TOKEN   Value: hf_xxxxx...
from google.colab import userdata
import os
HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
print(f"Token loaded: {HF_TOKEN[:8]}...")
"""

CELL3 = '''\
# Cell 3: Inline R2E environment (no local repo needed)
import random, hashlib, re
from dataclasses import dataclass, field
from typing import Dict, Tuple

TASK_MAX_STEPS = {"easy": 40, "medium": 60, "hard": 90}
VALID_ACTIONS = [
    "insert", "adjust_left", "adjust_right", "increase_force",
    "probe_friction", "probe_alignment", "probe_stiffness", "commit_solution",
]

@dataclass
class Hidden:
    friction: str
    alignment: str
    stiffness: str

@dataclass
class State:
    position: float = 0.0
    force_feedback: float = 0.3
    lateral_instability: float = 0.1
    failure_signal: str = "none"
    last_action: str = "none"
    step_count: int = 0
    phase: str = "investigation"
    known: Dict[str, str] = field(default_factory=dict)
    done: bool = False

def sample_hidden(seed: int, task: str) -> Hidden:
    def rng(v):
        h = hashlib.sha256(f"{seed}:{v}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))
    f = rng("friction").choice(["low", "high"]) if task in ("easy", "medium", "hard") else "low"
    a = rng("alignment").choice(["aligned", "misaligned"]) if task in ("medium", "hard") else "aligned"
    s = rng("stiffness").choice(["rigid", "compliant"]) if task == "hard" else "rigid"
    return Hidden(friction=f, alignment=a, stiffness=s)

def get_phase(step: int, max_steps: int) -> str:
    r = step / max_steps
    if r < 0.4: return "investigation"
    if r < 0.7: return "verification"
    return "execution"

def env_step(st: State, h: Hidden, action: str, max_steps: int):
    r = -0.01
    if action == "probe_friction":
        if "friction" not in st.known:
            st.force_feedback = 0.85 if h.friction == "high" else 0.15
            st.known["friction"] = h.friction; r += 0.05
        else:
            r -= 0.05
    elif action == "probe_alignment":
        if "alignment" not in st.known:
            st.lateral_instability = 0.7 if h.alignment == "misaligned" else 0.1
            st.known["alignment"] = h.alignment; r += 0.05
        else:
            r -= 0.05
    elif action == "probe_stiffness":
        if "stiffness" not in st.known:
            if h.stiffness == "compliant":
                st.lateral_instability = min(1.0, st.lateral_instability + 0.4)
            else:
                st.lateral_instability = max(0.0, st.lateral_instability - 0.2)
            st.known["stiffness"] = h.stiffness; r += 0.05
        else:
            r -= 0.05
    elif action in ("adjust_left", "adjust_right"):
        if h.alignment == "misaligned":
            st.lateral_instability = max(0.0, st.lateral_instability - 0.3); r += 0.02
        else:
            r -= 0.05
    elif action == "insert":
        if st.lateral_instability > 0.5:
            st.failure_signal = "unstable"; r = -1.0; st.done = True
        else:
            d = 0.10 if h.friction == "high" else 0.20
            st.position = min(1.0, st.position + d); r += 0.1 * d
    elif action == "increase_force":
        if h.friction == "high" and h.stiffness == "compliant":
            st.failure_signal = "jam"; r = -1.0; st.done = True
        elif st.lateral_instability > 0.5:
            st.failure_signal = "unstable"; r = -1.0; st.done = True
        else:
            st.position = min(1.0, st.position + 0.40); r += 0.04
    elif action == "commit_solution":
        if st.position >= 1.0 and st.failure_signal == "none":
            r = 1.0; st.done = True
        else:
            r = -1.0; st.done = True
    else:
        r -= 0.05
    st.step_count += 1
    st.last_action = action
    st.phase = get_phase(st.step_count, max_steps)
    if st.step_count >= max_steps and not st.done:
        st.done = True
    return st, r, st.done

def oracle_act(st: State, h: Hidden) -> Tuple[str, str]:
    k = st.known
    if "friction" not in k:
        return "probe_friction", "I need to measure friction level before applying any force."
    if "alignment" not in k:
        return "probe_alignment", f"Friction={k.get('friction')}. Now checking alignment to assess wobble risk."
    if "stiffness" not in k:
        return "probe_stiffness", f"Friction={k.get('friction')}, alignment={k.get('alignment')}. Checking stiffness to assess JAM risk."
    if st.lateral_instability > 0.3:
        return "adjust_left", f"lateral_instability={st.lateral_instability:.2f} > 0.3. Must correct alignment before inserting."
    if st.position >= 1.0:
        return "commit_solution", f"position={st.position:.2f} has reached target 1.0. No failure signal. Committing."
    if h.friction == "high" and h.stiffness == "compliant":
        return "insert", "DANGER: friction=HIGH and stiffness=COMPLIANT detected. increase_force would cause JAM. Using safe insert."
    return "increase_force", f"Conditions safe: friction={h.friction}, stiffness={h.stiffness}. Using increase_force for efficiency."

print("R2E environment defined inline.")
'''

CELL4 = """\
# Cell 4: Load Qwen2.5-1.5B with Unsloth 4-bit quantization
import os
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    token=HF_TOKEN,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print(f"Model loaded: {MODEL_NAME}")
"""

CELL5 = '''\
# Cell 5: Generate oracle training dataset (self-contained, no external imports)
SYSTEM_PROMPT = (
    "You are an AI agent debugging a robotic arm precision insertion task.\\n"
    "Hidden properties: friction_level, alignment_error, stiffness.\\n"
    "Always think step-by-step before every action.\\n\\n"
    "Response format:\\n"
    "<think>\\n[your reasoning]\\n</think>\\n"
    "Action: [one of: insert, adjust_left, adjust_right, increase_force, "
    "probe_friction, probe_alignment, probe_stiffness, commit_solution]\\n\\n"
    "CRITICAL: NEVER use increase_force if friction=HIGH and stiffness=COMPLIANT (causes JAM)."
)

def make_user_msg(st: State, task: str) -> str:
    known_str = ", ".join(f"{k}={v}" for k, v in st.known.items()) or "none yet"
    return (
        f"Step {st.step_count + 1} | Task: {task} | Phase: {st.phase}\\n\\n"
        f"Observation:\\n"
        f"  position:            {st.position:.3f}\\n"
        f"  force_feedback:      {st.force_feedback:.3f}\\n"
        f"  lateral_instability: {st.lateral_instability:.3f}\\n"
        f"  failure_signal:      {st.failure_signal}\\n"
        f"  last_action:         {st.last_action}\\n\\n"
        f"Known properties: {known_str}\\n\\n"
        "Think and choose your action:"
    )

def collect_dataset(n_easy=80, n_medium=60, n_hard=40):
    prompts = []
    for task, n in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]:
        max_steps = TASK_MAX_STEPS[task]
        for seed in range(n):
            h = sample_hidden(seed, task)
            st = State()
            while not st.done:
                user_msg = make_user_msg(st, task)
                action, reasoning = oracle_act(st, h)
                full_prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts.append({
                    "prompt": full_prompt,
                    "target_action": action,
                    "task": task,
                    "seed": seed,
                })
                st, _, _ = env_step(st, h, action, max_steps)
        count = sum(1 for p in prompts if p["task"] == task)
        print(f"  Collected {task}: {count} prompts")
    return prompts

print("Collecting oracle episodes...")
prompts = collect_dataset()
print(f"Total: {len(prompts)} training prompts")
'''

CELL6 = """\
# Cell 6: Build HuggingFace Dataset
from datasets import Dataset
ds = Dataset.from_list(prompts).shuffle(seed=42)
split = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
"""

CELL7 = '''\
# Cell 7: GRPO reward functions
import re

def parse_response(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    reasoning = m.group(1).strip() if m else ""
    am = re.search(r"Action:\\s*([a-z_]+)", text, re.IGNORECASE)
    if am and am.group(1).lower() in VALID_ACTIONS:
        return reasoning, am.group(1).lower()
    for a in VALID_ACTIONS:
        if a in text.lower():
            return reasoning, a
    return reasoning, "probe_friction"

def format_reward(completions, **kwargs):
    """Reward correct output format: <think>...</think> + Action: <valid>"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        r = 0.0
        if re.search(r"<think>.*?</think>", text, re.DOTALL): r += 0.4
        if re.search(r"Action:\\s*[a-z_]+", text, re.IGNORECASE): r += 0.3
        _, a = parse_response(text)
        if a in VALID_ACTIONS: r += 0.3
        rewards.append(r)
    return rewards

def reasoning_reward(completions, **kwargs):
    """Reward causal reasoning about physical properties."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        reasoning, action = parse_response(text)
        t = reasoning.lower(); r = 0.0
        if "probe" in t: r += 0.15
        if "friction" in t: r += 0.10
        if "stiffness" in t or "compliant" in t: r += 0.10
        if any(w in t for w in ["jam", "danger", "risky", "avoid"]) and "increase_force" in t:
            r += 0.20
        if action == "commit_solution" and "position" not in t: r -= 0.30
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards

def safety_reward(completions, **kwargs):
    """Strong penalty for choosing increase_force in dangerous context."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        t = text.lower()
        _, action = parse_response(text)
        dangerous = ("friction=high" in t or "friction: high" in t) and "compliant" in t
        if dangerous and action == "increase_force":
            rewards.append(-1.0)
        elif action in VALID_ACTIONS:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards

print("Reward functions defined.")
'''

CELL8 = '''\
# Cell 8: GRPO Training (runtime API detection for cross-version compatibility)
from trl import GRPOTrainer, GRPOConfig
import inspect

grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
print(f"Detected {len(grpo_params)} GRPOConfig parameters in installed TRL version.")

# Build kwargs compatible with whichever TRL version is installed
kwargs = dict(
    output_dir="r2e_grpo_output",
    num_generations=4,
    temperature=0.8,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=100,
    report_to="none",
    fp16=True,
    seed=42,
)

# Completion length param name changed in TRL 0.11
if "max_completion_length" in grpo_params:
    kwargs["max_completion_length"] = 300
elif "max_new_tokens" in grpo_params:
    kwargs["max_new_tokens"] = 300

# Eval strategy param renamed in TRL 0.10
if "eval_strategy" in grpo_params:
    kwargs["eval_strategy"] = "steps"
elif "evaluation_strategy" in grpo_params:
    kwargs["evaluation_strategy"] = "steps"

if "eval_steps" in grpo_params:
    kwargs["eval_steps"] = 50

print(f"Training config: {list(kwargs.keys())}")
training_args = GRPOConfig(**kwargs)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, reasoning_reward, safety_reward],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Starting GRPO training...")
trainer.train()
print("Training complete!")
'''

CELL9 = '''\
# Cell 9: Plot training curves and save logs
import matplotlib.pyplot as plt, json

log = trainer.state.log_history
steps   = [x["step"] for x in log if "loss" in x]
rewards = [x.get("reward", x.get("train_reward", 0.0)) for x in log if "loss" in x]
losses  = [x["loss"] for x in log if "loss" in x]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(steps, rewards, color="#2ecc71", linewidth=2)
a1.set_title("GRPO Reward Curve\\nR2E-RoboLab v2 — Qwen2.5-1.5B")
a1.set_xlabel("Training Step"); a1.set_ylabel("Mean Reward")
a1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 baseline")
a1.legend(); a1.grid(alpha=0.3)

a2.plot(steps, losses, color="#e74c3c", linewidth=2)
a2.set_title("Training Loss\\nR2E-RoboLab v2 — Qwen2.5-1.5B")
a2.set_xlabel("Training Step"); a2.set_ylabel("Loss")
a2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

with open("training_log.json", "w") as f:
    json.dump(log, f, indent=2)
print("Saved: training_curves.png, training_log.json")
print(f"Final reward: {rewards[-1]:.4f}" if rewards else "No reward data found.")
'''

CELL10 = '''\
# Cell 10: Test trained model on deceptive hard case
TEST = (
    "Step 3 | Task: hard | Phase: investigation\\n\\n"
    "Observation:\\n"
    "  position:            0.000\\n"
    "  force_feedback:      0.850\\n"
    "  lateral_instability: 0.100\\n"
    "  failure_signal:      none\\n"
    "  last_action:         probe_friction\\n\\n"
    "Known properties: friction=high\\n\\n"
    "Think and choose your action:"
)
msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": TEST}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

FastLanguageModel.for_inference(model)
import torch
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=" * 60)
print("CONTEXT: friction=HIGH → increase_force would JAM the robot!")
print("EXPECTED: probe_stiffness or insert (NOT increase_force)")
print("=" * 60)
print(resp)
print("=" * 60)
'''

CELL11 = """\
# Cell 11: Push trained model to HuggingFace Hub
from huggingface_hub import login
login(token=HF_TOKEN)

REPO_ID = "monika-10333/r2e-robolab-qwen2.5-1.5b-grpo"
model.save_pretrained("r2e_trained_model")
tokenizer.save_pretrained("r2e_trained_model")
model.push_to_hub(REPO_ID, token=HF_TOKEN)
tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
print(f"Model pushed to: https://huggingface.co/{REPO_ID}")
"""

CELL12 = """\
# Cell 12: Download artifacts to your local machine
from google.colab import files
files.download("training_curves.png")
files.download("training_log.json")
print("Download initiated! Save both files to results/ in your local repo.")
"""

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": [
        md("# R2E-RoboLab v2 — GRPO Training Notebook\n"
           "**Fully self-contained** — no local repo imports needed.\n\n"
           "### Setup\n"
           "1. Runtime → Change runtime type → **T4 GPU**\n"
           "2. Click 🔑 in left sidebar → Add secret: `HF_TOKEN` = your `hf_...` token\n"
           "3. Runtime → Run all\n\n"
           "Expected time: ~90 min on T4"),
        cell(CELL1),
        cell(CELL2),
        cell(CELL3),
        cell(CELL4),
        cell(CELL5),
        cell(CELL6),
        cell(CELL7),
        cell(CELL8),
        cell(CELL9),
        cell(CELL10),
        cell(CELL11),
        cell(CELL12),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("train_r2e_grpo.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("train_r2e_grpo.ipynb written successfully.")
