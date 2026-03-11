from typing import Optional, Dict, Any

from pydantic import BaseModel


class CreateQuery(BaseModel):
    id: int = 0


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    score: float
    done: bool
    info: Optional[Dict[str, Any]] = None


class ResetQuery(BaseModel):
    env_idx: int
    id: Optional[int] = None
    split: Optional[str] = None
    seed: Optional[int] = None


class CloseRequestBody(BaseModel):
    env_idx: int
