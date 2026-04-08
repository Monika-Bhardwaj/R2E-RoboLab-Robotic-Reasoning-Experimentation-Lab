---
title: R2E RoboLab
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# R2E: Robotic Reasoning & Experimentation Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/open-env/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/monika-10333/r2e-robolab)

**R2E-RoboLab** is a production-quality OpenEnv environment that evaluates AI agents on **Active Experimentation** and **Reasoning under Uncertainty** — modelled on real-world industrial robotic manipulation debugging.

---

## 🎯 Motivation

In industrial robotics, debugging precision insertion failures is a major bottleneck. Root causes are hidden:
- High **friction** on mating surfaces slows progress
- A **misalignment** in the assembly jig causes lateral instability
- **Compliant stiffness** creates catastrophic jam risk under force

A skilled engineer does not blindly retry — they **probe**, **infer**, and **adapt**. R2E-RoboLab tests exactly this reasoning capability.

## 🔥 Key Differentiator: Deceptive Reward

The action `increase_force` gives immediate progress (reward +0.04), but triggers a terminal `jam` failure if the agent hasn't first verified safe physical conditions. Greedy agents fail; reasoning agents succeed.

---

## 📊 Environment Specification

### Hidden State (sampled from seed)
| Variable | Values |
|---|---|
| `friction_level` | `low` / `high` |
| `alignment_error` | `aligned` / `misaligned` |
| `stiffness` | `rigid` / `compliant` |

### Observation Space
| Field | Type | Range | Description |
|---|---|---|---|
| `position` | float | [0, 1] | Insertion depth |
| `force_feedback` | float | [0, 1] | Resistance signal |
| `lateral_instability` | float | [0, 1] | Wobble indicator |
| `failure_signal` | string | `none/jam/slip/unstable` | Current failure mode |
| `last_action` | string | — | Most recent action |
| `step_count` | int | — | Steps taken |

### Action Space
| Category | Action | Effect |
|---|---|---|
| Task | `insert` | Progress (+0.10 or +0.20 by friction) |
| Task | `adjust_left` / `adjust_right` | Corrects misalignment |
| Task | `increase_force` | Fast progress but **RISKY** |
| Probe | `probe_friction` | Reveals friction via `force_feedback` |
| Probe | `probe_alignment` | Reveals alignment via `lateral_instability` |
| Probe | `probe_stiffness` | Reveals stiffness via instability response |
| Meta | `commit_solution` | Terminal (requires `position=1.0`) |

---

## 🧪 Tasks & Benchmark Scores

| Task | Difficulty | Max Steps | Agent Score |
|---|---|---|---|
| `easy` | Single variable (friction only) | 30 | **0.78** |
| `medium` | Two variables (friction + alignment) | 40 | **0.82** |
| `hard` | All variables + deceptive rewards | 60 | **0.70** |

### Scoring Formula
```
score = 0.5 × success + 0.3 × efficiency + 0.2 × correctness
```

---

## 🌐 Live API

**Space URL:** https://huggingface.co/spaces/monika-10333/r2e-robolab

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/docs` | GET | Interactive Swagger UI |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Inspect current state |
| `/health` | GET | Health check |

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference
```bash
export HF_TOKEN="your_hf_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker
```bash
docker build -t r2e-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token r2e-env
```

---

## 📁 Project Structure

```
r2e-robolab/
├── r2e_env/          # Core environment (models, dynamics, environment)
├── server/           # FastAPI server (app.py)
├── tasks/            # Task configurations (easy, medium, hard)
├── graders/          # Deterministic graders per task
├── tests/            # Smoke tests & property-based tests
├── inference.py      # Baseline agent script (OpenAI-compatible)
├── openenv.yaml      # OpenEnv specification
├── Dockerfile        # Container definition
└── requirements.txt  # Runtime dependencies
```

## 📜 OpenEnv Compliance

Validated with `openenv validate`:
```
[OK] R2E-RoboLab: Ready for multi-mode deployment
```
