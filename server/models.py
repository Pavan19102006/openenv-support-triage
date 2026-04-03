"""Typed Pydantic models for the Ticket Triage OpenEnv environment."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    SHIPPING = "shipping"
    SECURITY = "security"
    GENERAL = "general"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    CLOSED = "closed"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class ActionType(str, Enum):
    CATEGORIZE = "categorize"
    SET_PRIORITY = "set_priority"
    RESPOND = "respond"
    ESCALATE = "escalate"
    ADD_NOTE = "add_note"
    CLOSE = "close"
    NOOP = "noop"


# ── Ticket ───────────────────────────────────────────────────────────────────

class Ticket(BaseModel):
    """A customer support ticket visible to the agent."""
    id: str
    subject: str
    body: str
    sender: str = "customer"
    sentiment: Sentiment = Sentiment.NEUTRAL
    status: TicketStatus = TicketStatus.OPEN
    category: Optional[str] = None
    priority: Optional[str] = None
    has_response: bool = False
    response_text: Optional[str] = None
    escalated: bool = False
    notes: List[str] = Field(default_factory=list)
    created_at: str = "2026-04-03T00:00:00Z"


# ── Ground-truth (internal only, never sent to agent) ────────────────────────

class TicketGroundTruth(BaseModel):
    """Internal ground truth for grading — never exposed to the agent."""
    id: str
    subject: str
    body: str
    sender: str = "customer"
    sentiment: Sentiment = Sentiment.NEUTRAL
    created_at: str = "2026-04-03T00:00:00Z"

    # Ground truth fields
    gt_category: TicketCategory
    gt_priority: TicketPriority
    gt_needs_response: bool = False
    gt_needs_escalation: bool = False
    gt_response_keywords: List[str] = Field(default_factory=list)

    # Mutable runtime state
    category: Optional[str] = None
    priority: Optional[str] = None
    status: TicketStatus = TicketStatus.OPEN
    has_response: bool = False
    response_text: Optional[str] = None
    escalated: bool = False
    notes: List[str] = Field(default_factory=list)


# ── Action / Observation / State ─────────────────────────────────────────────

class Action(BaseModel):
    """An action the agent can take."""
    action_type: str = Field(
        ...,
        description="One of: categorize, set_priority, respond, escalate, add_note, close, noop"
    )
    ticket_id: str = Field(default="", description="Target ticket ID (e.g. T1)")
    payload: str = Field(
        default="",
        description="Action argument: category name, priority level, response text, or note text"
    )


class Observation(BaseModel):
    """What the agent can observe."""
    tickets: List[Ticket]
    last_feedback: str = "Environment ready."
    current_score: float = 0.0
    pending_count: int = 0
    resolved_count: int = 0
    escalated_count: int = 0
    step_number: int = 0
    max_steps: int = 20


class State(BaseModel):
    """Full environment state for the state() endpoint."""
    task_id: str = "task_easy"
    step_count: int = 0
    max_steps: int = 20
    score: float = 0.0
    done: bool = False
    episode_reward: float = 0.0


# ── Request / Response wrappers ──────────────────────────────────────────────

class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)
