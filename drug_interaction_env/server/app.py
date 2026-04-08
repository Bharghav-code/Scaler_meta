"""
app.py — FastAPI server exposing the Drug Interaction Checker environment.

Endpoints:
  POST /reset  → Initialize a new episode
  POST /step   → Submit one agent action
  GET  /state  → Return current episode state
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Literal, Optional

from drug_interaction_env.server.drug_interaction_environment import DrugInteractionEnvironment

app = FastAPI(
    title="Drug Interaction Checker — OpenEnv",
    description="Clinical pharmacist simulator for drug interaction detection",
    version="1.0.0",
)


env = DrugInteractionEnvironment()
class ResetRequest(BaseModel):
    task_level: Literal["easy", "medium", "hard"] = "easy"


class StepRequest(BaseModel):
    action_type: Literal["flag_interaction", "DONE"]
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    suggested_action: Optional[Literal["monitor", "reduce_dose", "replace_drug"]] = None


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    state: dict
# API ENDPOINTS



@app.post("/reset")
def reset_endpoint(request: Optional[ResetRequest] = Body(default=None)):
    try:
        if request is None:
            request = ResetRequest()
        observation = env.reset(request.task_level)
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step_endpoint(request: StepRequest):
    """Submit one agent action and receive reward + observation."""
    try:
        action = request.model_dump()
        observation, reward, done, state = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            state=state,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state_endpoint():
    """Return current episode state."""
    if env.patient is None:
        raise HTTPException(status_code=400, detail="No episode started. Call /reset first.")
    
    state_dict = env.state
    state_dict["episode_score"] = env.get_episode_score()
    return state_dict


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
