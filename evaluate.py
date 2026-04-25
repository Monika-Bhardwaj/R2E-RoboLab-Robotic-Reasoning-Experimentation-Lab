"""
evaluate.py — Multi-seed evaluation of all 5 agent baselines.

Runs each agent on 50 seeds × 3 tasks = 750 episodes.
Outputs a comparison table and saves plots to results/.

Usage:
    python evaluate.py
    python evaluate.py --seeds 10 --tasks easy hard
"""
import asyncio
import argparse
import json
import os
import statistics
from pathlib import Path

from r2e_env.environment import R2EEnv
from r2e_env.models import R2EAction
from agents import (
    RandomAgent, GreedyAgent, DeterministicAgent,
    ReasoningAgent, OracleAgent,
)
from graders import get_grader

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


async def run_episode(agent, task: str, seed: int) -> dict:
    """Run a single episode and return score details."""
    env = R2EEnv()
    grader = get_grader(task)
    probed = set()

    obs = await env.reset(task=task, seed=seed)
    done = False

    while not done:
        # Get hidden state for oracle agent
        hidden = env._hidden if agent.name == "oracle" else None
        action = agent.act(obs, probed, hidden=hidden)
        obs, reward, done, info = await env.step(action)

    log = env.get_episode_log()
    result = grader.grade(log)
    return {
        "task": task,
        "seed": seed,
        "agent": agent.name,
        "score": result.score,
        "success": result.success,
        "efficiency": result.efficiency,
        "correctness": result.correctness,
        "reasoning_score": result.reasoning_score,
        "steps": len(log.steps),
        "revealed_probes": log.revealed_probes,
    }


async def evaluate_agent(agent, tasks: list[str], seeds: list[int]) -> dict:
    """Evaluate an agent across all tasks and seeds."""
    all_results = []
    total = len(tasks) * len(seeds)
    done_count = 0

    for task in tasks:
        for seed in seeds:
            result = await run_episode(agent, task, seed)
            all_results.append(result)
            done_count += 1
            if done_count % 10 == 0:
                print(f"  [{agent.name}] {done_count}/{total} episodes done...")

    return all_results


def summarize(results: list[dict], agent_name: str, task: str) -> dict:
    task_results = [r for r in results if r["task"] == task]
    if not task_results:
        return {}
    scores = [r["score"] for r in task_results]
    successes = [r["success"] for r in task_results]
    return {
        "agent": agent_name,
        "task": task,
        "mean_score": round(statistics.mean(scores), 4),
        "std_score": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        "success_rate": round(statistics.mean(successes), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "n_episodes": len(scores),
    }


def print_table(summaries: list[dict], tasks: list[str]):
    print("\n" + "=" * 90)
    print(f"{'Agent':<20} {'Task':<10} {'Mean Score':<14} {'Std':<10} {'Success Rate':<15} {'N'}")
    print("=" * 90)
    for s in summaries:
        print(f"{s['agent']:<20} {s['task']:<10} {s['mean_score']:<14.4f} "
              f"{s['std_score']:<10.4f} {s['success_rate']:<15.2%} {s['n_episodes']}")
    print("=" * 90)


def save_results(all_data: dict, summaries: list[dict]):
    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(all_data, f, indent=2)
    with open(RESULTS_DIR / "summary_table.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


def plot_results(summaries: list[dict], tasks: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        agents = list(dict.fromkeys(s["agent"] for s in summaries))
        colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"]

        for task in tasks:
            task_data = [s for s in summaries if s["task"] == task]
            means = [s["mean_score"] for s in task_data]
            stds = [s["std_score"] for s in task_data]
            agent_names = [s["agent"] for s in task_data]

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(agent_names))
            bars = ax.bar(x, means, yerr=stds, capsize=6,
                          color=colors[:len(agent_names)], alpha=0.85, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(agent_names, fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Mean Score ± Std", fontsize=12)
            ax.set_title(f"Agent Comparison — {task.capitalize()} Task\n"
                         f"(R2E-RoboLab v2, n={task_data[0]['n_episodes'] if task_data else 0} seeds)",
                         fontsize=13)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 baseline")
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            ax.legend()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"comparison_{task}.png", dpi=150)
            plt.close()
            print(f"Plot saved: results/comparison_{task}.png")

        # Success rate comparison across all tasks
        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
        if len(tasks) == 1:
            axes = [axes]
        for ax, task in zip(axes, tasks):
            task_data = [s for s in summaries if s["task"] == task]
            rates = [s["success_rate"] for s in task_data]
            names = [s["agent"] for s in task_data]
            ax.bar(names, rates, color=colors[:len(names)], alpha=0.85, edgecolor="black")
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{task.capitalize()}", fontsize=12)
            ax.set_ylabel("Success Rate" if task == tasks[0] else "")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Success Rate by Agent and Task — R2E-RoboLab v2", fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "success_rate_comparison.png", dpi=150)
        plt.close()
        print("Plot saved: results/success_rate_comparison.png")

    except ImportError:
        print("matplotlib not installed — skipping plots. Run: pip install matplotlib")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds per task")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    tasks = args.tasks

    agents = [
        RandomAgent(),
        GreedyAgent(),
        DeterministicAgent(),
        ReasoningAgent(),
        OracleAgent(),
    ]

    print(f"Evaluating {len(agents)} agents × {len(tasks)} tasks × {len(seeds)} seeds "
          f"= {len(agents) * len(tasks) * len(seeds)} episodes\n")

    all_data = {}
    summaries = []

    for agent in agents:
        print(f"\nRunning {agent.name} agent...")
        results = await evaluate_agent(agent, tasks, seeds)
        all_data[agent.name] = results
        for task in tasks:
            s = summarize(results, agent.name, task)
            if s:
                summaries.append(s)

    print_table(summaries, tasks)
    save_results(all_data, summaries)
    plot_results(summaries, tasks)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
