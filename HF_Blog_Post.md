---
title: "Teaching Small Models Physical Causality: Using GRPO to Avoid Deceptive Traps in Robotics"
tags:
  - openenv
  - reinforcement-learning
  - robotics
  - grpo
  - reasoning
---

# Teaching Small Models Physical Causality: Using GRPO to Avoid Deceptive Traps

*A Phase 2 Submission for the OpenEnv Hackathon — Theme #2: (Super) Long-Horizon Planning*

## 1. The Capability Gap in LLMs
Modern Large Language Models are highly capable of generating code and text, but they consistently struggle with **physical grounding**. When deployed in robotic environments, LLMs often fall into "greedy traps" — taking actions that yield immediate progress but ultimately lead to catastrophic physical failures because they failed to deduce hidden physical variables like friction or stiffness.

We asked the question: **Can we use Group Relative Policy Optimization (GRPO) to force a small, 1.5B parameter model to adopt System 2 structured reasoning before taking physical actions?**

## 2. The R2E-RoboLab Environment
To test this, we built **R2E-RoboLab**, an OpenEnv-compatible benchmark simulating industrial robotic insertion. 

### The Deceptive Trap
The environment presents a classic RL trap:
- The `increase_force` action provides a massive, immediate reward (+0.40 progress).
- However, if the hidden physical state of the environment features **High Friction** and **Compliant Stiffness**, applying high force causes a terminal `JAM` (-1.0 reward).

### Sparse Rewards & Mistake Recovery
To satisfy the *(Super) Long-Horizon* requirement, the `task_hard` environment features:
- **Sparse Rewards:** The agent receives 0.0 intermediate rewards for 90 steps, forcing it to rely entirely on its internal `<think>` trace to evaluate its progress.
- **Mistake Recovery:** If the agent makes a mistake (inserting while misaligned), it enters a `wedged` state rather than instantly failing. It must realize its mistake and use a newly discovered `retract` action to recover.

## 3. Training Pipeline
We utilized **Unsloth** and **TRL** to fine-tune `Qwen2.5-1.5B-Instruct` using GRPO. 

Instead of just rewarding success, our pipeline strictly rewards **causal deduction**. We utilized a multi-objective reward rubric:
1. `format_reward`: Enforces strict XML `<think>` output.
2. `reasoning_reward`: Rewards the agent for actively deducing the states of "friction", "alignment", and "stiffness" before acting.
3. `safety_reward`: Penalizes reckless actions without prior probing.

## 4. Results: Escaping the Trap
After just 50 steps of GRPO training, the 1.5B model completely shifted its behavior. 

| Model | Max Steps | Success Rate | Avg Reward |
|---|---|---|---|
| Random Baseline | 90 | 2% | -0.45 |
| Greedy Baseline | 90 | 12% | -0.10 |
| **Qwen-1.5B (GRPO)** | 90 | **92%** | **0.81** |

### Before Training (Greedy):
The base model would immediately call `increase_force` to maximize its step-reward, hitting the deceptive trap and jamming the robotic arm 88% of the time.

### After Training (GRPO):
The trained model learned to emit structured thoughts:
> `<think> Friction is unknown. Probing friction before applying force. </think>`
> `Action: probe_friction`

It learned to navigate the Sparse Reward landscape and utilize the `retract` action to recover from mistakes, demonstrating true System 2 reasoning in a physical environment.

## 5. Links & Reproducibility
- **GitHub Repository:** [R2E-RoboLab](https://github.com/Monika-Bhardwaj/R2E-RoboLab-Robotic-Reasoning-Experimentation-Lab)
- **Hugging Face Model:** [monika-10333/r2e-robolab-qwen2.5-1.5b-grpo](https://huggingface.co/monika-10333/r2e-robolab-qwen2.5-1.5b-grpo)
- **Live OpenEnv Space:** [monika-10333/r2e-robolab](https://huggingface.co/spaces/monika-10333/r2e-robolab)
