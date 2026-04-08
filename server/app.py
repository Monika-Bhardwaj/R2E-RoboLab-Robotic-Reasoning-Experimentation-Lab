from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from r2e_env.environment import R2EEnv
from r2e_env.models import R2EAction

app = FastAPI(
    title="R2E-RoboLab",
    description="Robotic Reasoning & Experimentation Lab — OpenEnv Environment",
    version="1.0.0",
)

env = R2EEnv()


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action: str = "probe_friction"


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>R2E-RoboLab</title></head>
    <body style="font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px">
        <h1>R2E-RoboLab</h1>
        <p><strong>Robotic Reasoning and Experimentation Environment</strong></p>
        <ul>
            <li><code>POST /reset</code> Start a new episode</li>
            <li><code>POST /step</code> Take an action</li>
            <li><code>GET /state</code> Inspect current state</li>
            <li><a href="/docs">API Docs</a></li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


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
    action = R2EAction(action=request.action)
    obs, reward, done, info = await env.step(action)
    return {
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
