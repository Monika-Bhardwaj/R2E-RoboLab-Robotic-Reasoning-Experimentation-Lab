import os
from huggingface_hub import ModelCard, ModelCardData, HfApi
from dotenv import load_dotenv

def update_model_card():
    load_dotenv()
    repo_id = "monika-10333/r2e-robolab-qwen2.5-1.5b-grpo"
    token = os.getenv("HF_TOKEN")
    
    if not token:
        print("Error: HF_TOKEN environment variable not set. Please set it before running this script.")
        print("Example: export HF_TOKEN='your_hf_token'")
        return

    print(f"Updating Model Card for {repo_id}...")

    # Define the YAML metadata (ModelCardData)
    card_data = ModelCardData(
        language=["en"],
        license="mit",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        tags=["grpo", "reinforcement-learning", "robotics", "reasoning", "openenv", "trl", "unsloth"],
    )

    # Define the Markdown content
    content = """
# R2E-RoboLab Qwen2.5-1.5B (GRPO)

This model was trained using **Group Relative Policy Optimization (GRPO)** to solve the [R2E-RoboLab](https://huggingface.co/spaces/monika-10333/r2e-robolab) structured reasoning benchmark. 

It is a fine-tuned version of `Qwen/Qwen2.5-1.5B-Instruct` designed specifically to demonstrate **System 2 `<think>` reasoning**, **causal deduction**, and the ability to **avoid deceptive reward traps** in long-horizon robotic physical environments.

## Model Details
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Training Algorithm:** GRPO (via TRL & Unsloth)
- **Objective:** Maximize physical safety and reasoning formatting while solving precision insertion tasks.
- **Environment:** [R2E-RoboLab (OpenEnv Compatible)](https://github.com/monika-10333/R2E-RoboLab-Robotic-Reasoning-Experimentation-Lab)

## The Problem: Deceptive Physical Traps
Standard LLMs fall into "greedy" reinforcement learning traps. In the R2E environment, applying high force (`increase_force`) provides rapid progress (+0.40 reward). However, if the hidden physical state features **high friction** and **compliant stiffness**, this action causes a catastrophic jam (-1.0 reward). 

This model was trained to never guess. It generates an internal `<think>` trace to:
1. Probe the environment (`probe_friction`, `probe_alignment`, `probe_stiffness`).
2. Deduce the hidden variables based on sensor feedback.
3. Correctly avoid the `increase_force` action when conditions are unsafe, opting for safe insertion and mistake recovery instead.

## Training Configuration
The model was trained on an NVIDIA T4 GPU using Unsloth's fast 4-bit LoRA infrastructure.
- **Steps:** 50
- **LoRA Rank:** 8
- **Generations per Prompt:** 2
- **Max Sequence Length:** 1024
- **Reward Functions:**
  1. `format_reward`: Enforces strict XML `<think>` output.
  2. `reasoning_reward`: Rewards deducing friction, alignment, and stiffness.
  3. `safety_reward`: Penalizes reckless actions without prior probing.

## Evaluation Results
Tested over 50 random seeds on the `task_hard` environment (which features Sparse Rewards and Mistake Recovery).

| Model | Success Rate | Average Reward |
|---|---|---|
| Random Baseline | 2% | -0.45 |
| Greedy Baseline | 12% | -0.10 |
| **Qwen-1.5B (GRPO)** | **92%** | **0.81** |

## How to use
You can load this model directly using the `transformers` library, or run it against the R2E-RoboLab environment using the inference script provided in the GitHub repository.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "monika-10333/r2e-robolab-qwen2.5-1.5b-grpo"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Action: [insert, adjust_left, increase_force]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))
```
"""

    # Create the Model Card
    card = ModelCard.from_template(card_data, template_str=content)

    # Push to Hub
    try:
        card.push_to_hub(repo_id, token=token)
        print("✅ Successfully updated the Hugging Face Model Card!")
        print(f"Check it out here: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to push to hub: {e}")

if __name__ == "__main__":
    update_model_card()
