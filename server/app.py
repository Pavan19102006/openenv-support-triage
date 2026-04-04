"""
FastAPI application for the Ticket Triage Environment.

Uses the OpenEnv framework's create_fastapi_app to generate the standard
API endpoints (reset, step, state, health, schema, ws) automatically.

Custom stateful endpoints are added on top for the test suite and
inference agent which need persistent state across HTTP calls.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TriageAction, TriageObservation
from server.triage_environment import TriageEnvironment

# ── Create the app using the OpenEnv framework ───────────────────────────
# create_fastapi_app wraps TriageEnvironment with standard OpenEnv endpoints:
#   POST /reset, POST /step, GET /state, GET /health, GET /schema, WS /ws

try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(TriageEnvironment, TriageAction, TriageObservation)
except ImportError:
    # Fallback if create_fastapi_app is not available in this version
    from fastapi import FastAPI
    app = FastAPI(
        title="Ticket Triage OpenEnv",
        description="IT Helpdesk Ticket Triage Environment for OpenEnv",
        version="1.0.0",
    )

# ── Stateful endpoints for test suite & inference agent ───────────────────
# The OpenEnv HTTP endpoints are stateless (each request creates a new env).
# For full episode interaction, we keep a persistent global env instance.

from pydantic import BaseModel
from typing import Optional

_env = TriageEnvironment()
_episode_reward = 0.0
_current_task_id = "task_easy"
_is_done = False


class _ResetReq(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class _ActionPayload(BaseModel):
    action_type: str = "noop"
    ticket_id: str = ""
    payload: str = ""


class _StepReq(BaseModel):
    action: _ActionPayload


@app.post("/stateful/reset")
def stateful_reset(request: _ResetReq = None):
    """Stateful reset — persists env state for subsequent /stateful/step calls."""
    global _episode_reward, _current_task_id, _is_done

    if request is None:
        request = _ResetReq()

    _episode_reward = 0.0
    _current_task_id = request.task_id
    _is_done = False

    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return {"observation": obs_dict, "reward": None, "done": False}


@app.post("/stateful/step")
def stateful_step(request: _StepReq):
    """Stateful step — uses the env persisted from /stateful/reset."""
    global _episode_reward, _is_done

    action = TriageAction(
        action_type=request.action.action_type,
        ticket_id=request.action.ticket_id,
        payload=request.action.payload,
    )
    obs = _env.step(action)

    reward = obs.reward if obs.reward is not None else 0.0
    done = obs.done
    _episode_reward += reward
    _is_done = done

    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return {"observation": obs_dict, "reward": reward, "done": done}


@app.get("/stateful/state")
def stateful_state():
    """Get current stateful environment state."""
    score = _env._compute_score()
    state_data = _env.state
    state_dict = state_data.model_dump() if hasattr(state_data, "model_dump") else {}
    state_dict["task_id"] = _current_task_id
    state_dict["max_steps"] = _env._max_steps
    state_dict["score"] = round(score, 4)
    state_dict["done"] = _is_done
    state_dict["episode_reward"] = round(_episode_reward, 4)
    return state_dict


@app.get("/")
def root():
    """Root endpoint with environment info."""
    return {
        "environment": "ticket-triage",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
    }


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()