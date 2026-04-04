"""
Data models for the Ticket Triage Environment.

Defines Action and Observation types used by the OpenEnv framework.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class TriageAction(Action):
    """Action for the Ticket Triage environment."""

    action_type: str = Field(
        ...,
        description="One of: categorize, set_priority, respond, escalate, add_note, close",
    )
    ticket_id: str = Field(default="", description="Target ticket ID (e.g. T1)")
    payload: str = Field(
        default="",
        description="Action payload - category name, priority level, or response text",
    )


class TriageObservation(Observation):
    """Observation from the Ticket Triage environment."""

    tickets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Current ticket states"
    )
    last_feedback: str = Field(
        default="", description="Feedback from the last action"
    )
    pending_count: int = Field(default=0, description="Number of pending tickets")
    resolved_count: int = Field(default=0, description="Number of resolved tickets")
    escalated_count: int = Field(default=0, description="Number of escalated tickets")
    step_number: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps for this task")
    current_score: float = Field(default=0.0, description="Current grader score 0-1")
