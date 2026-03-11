from typing import Optional

from fastapi import FastAPI, Request

from .env_wrapper import sokoban_env_server
from .model import CloseRequestBody, CreateQuery, ResetQuery, StepQuery, StepResponse

app = FastAPI()


@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    return await call_next(request)


@app.get("/", response_model=str)
def generate_ok():
    return "ok"


@app.post("/create", response_model=int)
def create(create_query: Optional[CreateQuery] = None):
    query = create_query or CreateQuery(id=0)
    return sokoban_env_server.create(query.id)


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    observation, reward, score, done, info = _unpack_step(
        sokoban_env_server.step(step_query.env_idx, step_query.action)
    )
    return StepResponse(
        observation=observation,
        reward=reward,
        score=score,
        done=done,
        info=info,
    )


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return sokoban_env_server.observation(env_idx)


@app.post("/reset")
def reset(reset_query: ResetQuery):
    return sokoban_env_server.reset(
        env_idx=reset_query.env_idx,
        item_id=reset_query.id,
        split=reset_query.split,
        seed=reset_query.seed,
    )


@app.post("/close")
def close(body: CloseRequestBody):
    return sokoban_env_server.close(body.env_idx)


def _unpack_step(step_resp):
    return (
        str(step_resp.get("observation", "")),
        float(step_resp.get("reward", 0.0)),
        float(step_resp.get("score", 0.0)),
        bool(step_resp.get("done", False)),
        step_resp.get("info", None),
    )
