"""
FastAPI application for the Ticket Triage Environment.

Custom server that wraps TriageEnvironment with the exact API
the OpenEnv checker and test suite expect.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from server.triage_environment import TriageEnvironment

# ── Global environment instance (persistent across requests) ──────────────

env = TriageEnvironment()
episode_reward = 0.0
current_task_id = "task_easy"
is_done = False

# ── Request / Response models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None

class ActionPayload(BaseModel):
    action_type: str = "noop"
    ticket_id: str = ""
    payload: str = ""

class StepRequest(BaseModel):
    action: ActionPayload

# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Ticket Triage OpenEnv",
    description="IT Helpdesk Ticket Triage Environment for OpenEnv",
    version="1.0.0",
)


@app.get("/")
def root():
    """Root endpoint with environment info."""
    return {
        "environment": "ticket-triage",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    """Reset the environment and return the initial observation.

    Returns the OpenEnv-standard ResetResponse format:
    {"observation": {...}, "reward": null, "done": false}
    """
    global episode_reward, current_task_id, is_done

    if request is None:
        request = ResetRequest()

    episode_reward = 0.0
    current_task_id = request.task_id
    is_done = False

    obs = env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )

    # Serialize observation excluding reward, done, metadata (standard OpenEnv format)
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})

    return {
        "observation": obs_dict,
        "reward": None,
        "done": False,
    }


@app.post("/step")
def step(request: StepRequest):
    """Execute an action and return observation + reward.

    Returns the OpenEnv-standard StepResponse format:
    {"observation": {...}, "reward": float, "done": bool}
    """
    global episode_reward, is_done

    from models import TriageAction
    action = TriageAction(
        action_type=request.action.action_type,
        ticket_id=request.action.ticket_id,
        payload=request.action.payload,
    )

    obs = env.step(action)

    reward = obs.reward if obs.reward is not None else 0.0
    done = obs.done
    episode_reward += reward
    is_done = done

    # Serialize observation excluding reward, done, metadata (standard OpenEnv format)
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


@app.get("/state")
def get_state():
    """Get current environment state."""
    score = env._compute_score()
    state_data = env.state
    if hasattr(state_data, "model_dump"):
        state_dict = state_data.model_dump()
    else:
        state_dict = {
            "episode_id": getattr(state_data, "episode_id", None),
            "step_count": getattr(state_data, "step_count", 0),
        }

    # Add extra info
    state_dict["task_id"] = current_task_id
    state_dict["max_steps"] = env._max_steps
    state_dict["score"] = round(score, 4)
    state_dict["done"] = is_done
    state_dict["episode_reward"] = round(episode_reward, 4)

    return state_dict


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()