import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from r2e_env.environment import R2EEnv
from r2e_env.models import R2EAction

app = FastAPI(
    title="R2E-RoboLab v2",
    description="Robotic Reasoning & Experimentation Lab — Structured Reasoning Environment",
    version="2.0.0",
)

env = R2EEnv()


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action: str = "probe_friction"
    reasoning: str = ""    # optional <think> content
    raw_response: str = "" # optional: full LLM response to auto-parse


def parse_llm_response(raw: str) -> tuple[str, str]:
    """Extract reasoning and action from a raw LLM response."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    reasoning = think_match.group(1).strip() if think_match else ""

    action_match = re.search(
        r"\bAction:\s*([a-z_]+)",
        raw, re.IGNORECASE
    )
    if not action_match:
        # Fallback: last word that matches a valid action
        valid = ["insert", "adjust_left", "adjust_right", "increase_force",
                 "probe_friction", "probe_alignment", "probe_stiffness", "commit_solution"]
        for action in reversed(valid):
            if action in raw.lower():
                return reasoning, action
        return reasoning, "probe_friction"

    action = action_match.group(1).strip().lower()
    return reasoning, action


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>R2E-RoboLab v2</title></head>
    <body style="font-family:sans-serif;max-width:760px;margin:40px auto;padding:0 20px;color:#1a1a2e">
        <h1>🤖 R2E-RoboLab <span style="font-size:0.6em;color:#666">v2.0 — Structured Reasoning</span></h1>
        <p><strong>Robotic Reasoning &amp; Experimentation Environment</strong></p>
        <p>An OpenEnv benchmark where agents must <em>think before they act</em> in robotic manipulation debugging.
        The environment rewards correct causal reasoning about hidden physical properties.</p>
        <h3>API Endpoints</h3>
        <ul>
            <li><code>POST /reset</code> — Start a new episode</li>
            <li><code>POST /step</code> — Take an action (with optional reasoning)</li>
            <li><code>POST /step_raw</code> — Send raw LLM output (auto-parses &lt;think&gt; + Action:)</li>
            <li><code>GET /state</code> — Full internal state</li>
            <li><code>GET /health</code> — Health check</li>
            <li><a href="/docs">📖 Interactive API Docs</a></li>
        </ul>
        <h3>Tasks</h3>
        <ul>
            <li><code>easy</code> — Friction only (40 steps)</li>
            <li><code>medium</code> — Friction + Alignment (60 steps)</li>
            <li><code>hard</code> — All variables + Deceptive Reward Trap (90 steps)</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()
    obs = await env.reset(task=request.task, seed=request.seed)
    return obs.model_dump()


@app.post("/step")
async def step(request: Optional[StepRequest] = None):
    if request is None:
        request = StepRequest()

    # If raw LLM response provided, auto-parse it
    reasoning = request.reasoning
    action_str = request.action
    if request.raw_response:
        reasoning, action_str = parse_llm_response(request.raw_response)

    action = R2EAction(action=action_str, reasoning=reasoning)
    obs, reward, done, info = await env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/step_raw")
async def step_raw(body: dict):
    """Accept raw LLM output string, auto-parse <think> and Action: fields."""
    raw = body.get("response", "")
    reasoning, action_str = parse_llm_response(raw)
    action = R2EAction(action=action_str, reasoning=reasoning)
    obs, reward, done, info = await env.step(action)
    return {
        "parsed_reasoning": reasoning[:200],
        "parsed_action": action_str,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state():
    return env.state()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
