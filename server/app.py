"""
Ticket Triage OpenEnv — FastAPI server.

Endpoints:
    POST /reset   — reset the environment to a task
    POST /step    — take an action
    GET  /state   — get current state
    GET  /health  — health check
"""

from fastapi import FastAPI
from typing import Dict, List, Optional
import copy
import uvicorn

from server.models import (
    Action, Observation, State, Ticket, TicketGroundTruth,
    StepRequest, StepResponse, ResetRequest, ResetResponse,
    TicketCategory, TicketPriority, TicketStatus, Sentiment,
)

app = FastAPI(title="Ticket Triage Environment", version="1.0.0")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS — ground-truth tickets for each difficulty level
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: Dict[str, List[dict]] = {
    # ── EASY: 2 straightforward tickets ──────────────────────────────────────
    "task_easy": [
        {
            "id": "T1",
            "subject": "Cannot log into my account",
            "body": (
                "Hi, I've been trying to log into my account for the past hour. "
                "Every time I enter my credentials, I get an 'Invalid password' error "
                "even though I'm sure my password is correct. I've tried clearing my "
                "browser cache and using a different browser but the issue persists. "
                "My username is john.doe@email.com. Please help!"
            ),
            "sender": "john.doe@email.com",
            "sentiment": "neutral",
            "created_at": "2026-04-03T08:15:00Z",
            "gt_category": "technical",
            "gt_priority": "medium",
            "gt_needs_response": True,
            "gt_needs_escalation": False,
            "gt_response_keywords": ["password", "reset", "account"],
        },
        {
            "id": "T2",
            "subject": "Request for refund on order #9921",
            "body": (
                "Hello, I placed order #9921 last week and the product arrived "
                "damaged. The screen has a crack running across it. I'd like a "
                "full refund please. I have photos of the damage if needed."
            ),
            "sender": "sarah.m@email.com",
            "sentiment": "negative",
            "created_at": "2026-04-03T09:30:00Z",
            "gt_category": "billing",
            "gt_priority": "medium",
            "gt_needs_response": True,
            "gt_needs_escalation": False,
            "gt_response_keywords": ["refund", "order", "damage"],
        },
    ],

    # ── MEDIUM: 3 tickets, one needs escalation ──────────────────────────────
    "task_medium": [
        {
            "id": "T3",
            "subject": "How do I reset my password?",
            "body": (
                "I forgot my password and need to reset it. I don't see a "
                "'Forgot Password' link on the login page. Can you walk me "
                "through the process? My account email is mike.j@email.com."
            ),
            "sender": "mike.j@email.com",
            "sentiment": "neutral",
            "created_at": "2026-04-03T07:00:00Z",
            "gt_category": "technical",
            "gt_priority": "low",
            "gt_needs_response": True,
            "gt_needs_escalation": False,
            "gt_response_keywords": ["password", "reset", "link"],
        },
        {
            "id": "T4",
            "subject": "Order #443 has not arrived",
            "body": (
                "I placed order #443 two weeks ago and it still hasn't arrived. "
                "The tracking number TRACK-8812 shows no updates since April 1st. "
                "This is very frustrating as it was a birthday gift."
            ),
            "sender": "emma.w@email.com",
            "sentiment": "negative",
            "created_at": "2026-04-03T10:45:00Z",
            "gt_category": "shipping",
            "gt_priority": "high",
            "gt_needs_response": True,
            "gt_needs_escalation": False,
            "gt_response_keywords": ["order", "tracking", "delivery"],
        },
        {
            "id": "T5",
            "subject": "I was charged twice for the same order!",
            "body": (
                "I just checked my bank statement and I see TWO charges of $149.99 "
                "for order #7788. I should only have been charged once! This is "
                "unacceptable. I need the duplicate charge reversed immediately. "
                "Transaction IDs: TXN-001122 and TXN-001123."
            ),
            "sender": "alex.r@email.com",
            "sentiment": "angry",
            "created_at": "2026-04-03T06:20:00Z",
            "gt_category": "billing",
            "gt_priority": "high",
            "gt_needs_response": True,
            "gt_needs_escalation": True,
            "gt_response_keywords": ["charge", "refund", "duplicate", "reversed"],
        },
    ],

    # ── HARD: 4 tickets with security, angry customer, mixed priorities ──────
    "task_hard": [
        {
            "id": "T6",
            "subject": "URGENT: My account has been compromised",
            "body": (
                "Someone has gained access to my account! I received an email "
                "notification about a login from an IP address in another country "
                "(IP: 185.220.101.45). My saved payment methods have been used to "
                "make unauthorized purchases totaling $2,340. I need my account "
                "frozen immediately! I have not shared my credentials with anyone."
            ),
            "sender": "david.k@email.com",
            "sentiment": "angry",
            "created_at": "2026-04-03T03:15:00Z",
            "gt_category": "security",
            "gt_priority": "critical",
            "gt_needs_response": False,
            "gt_needs_escalation": True,
            "gt_response_keywords": [],
        },
        {
            "id": "T7",
            "subject": "API v2 endpoint returning 500 errors",
            "body": (
                "Our integration with your API v2 has been failing since yesterday. "
                "The /api/v2/orders endpoint returns HTTP 500 with this error body: "
                '{"error": "internal_server_error", "trace_id": "abc-123-def"}. '
                "This is blocking our production pipeline. We've confirmed our API "
                "key is valid and rate limits are not exceeded. Request ID: REQ-9945."
            ),
            "sender": "devops@partner-corp.com",
            "sentiment": "negative",
            "created_at": "2026-04-03T04:30:00Z",
            "gt_category": "technical",
            "gt_priority": "high",
            "gt_needs_response": True,
            "gt_needs_escalation": True,
            "gt_response_keywords": ["api", "error", "investigating", "engineering"],
        },
        {
            "id": "T8",
            "subject": "This is the WORST service I have ever experienced!!!",
            "body": (
                "I am FURIOUS. I ordered a laptop 3 weeks ago, it arrived broken, "
                "I sent it back for replacement, and now you're telling me the "
                "replacement is OUT OF STOCK?! I want a FULL refund plus compensation "
                "for the time I've wasted. Order #5566. If this is not resolved TODAY "
                "I am contacting my lawyer and posting reviews everywhere. "
                "ABSOLUTELY UNACCEPTABLE."
            ),
            "sender": "furious.customer@email.com",
            "sentiment": "angry",
            "created_at": "2026-04-03T05:00:00Z",
            "gt_category": "billing",
            "gt_priority": "high",
            "gt_needs_response": True,
            "gt_needs_escalation": True,
            "gt_response_keywords": ["refund", "apologize", "sorry", "understand"],
        },
        {
            "id": "T9",
            "subject": "What are your business hours?",
            "body": (
                "Hi there, I'm trying to find your customer service phone number "
                "and business hours. I checked the website but couldn't find the "
                "information. Can you help?"
            ),
            "sender": "casual.user@email.com",
            "sentiment": "positive",
            "created_at": "2026-04-03T09:00:00Z",
            "gt_category": "general",
            "gt_priority": "low",
            "gt_needs_response": True,
            "gt_needs_escalation": False,
            "gt_response_keywords": ["hours", "contact", "phone"],
        },
    ],
}

# Max achievable reward per task (for normalization)
# Calculated as sum of all correct action rewards per ticket
TASK_MAX_SCORES: Dict[str, float] = {
    "task_easy":   1.0,   # 2 tickets × (cat 0.15 + pri 0.10 + resp 0.15 + close 0.15) = 1.10 → cap 1.0
    "task_medium":  1.0,  # 3 tickets, one with escalation
    "task_hard":    1.0,  # 4 tickets, complex mix
}

VALID_CATEGORIES = {"technical", "billing", "shipping", "security", "general"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
MAX_STEPS = {"task_easy": 20, "task_medium": 25, "task_hard": 30}


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT STATE (single-session)
# ═══════════════════════════════════════════════════════════════════════════════

class EnvState:
    """Mutable environment state for one episode."""

    def __init__(self):
        self.task_id: str = "task_easy"
        self.tickets: List[TicketGroundTruth] = []
        self.last_feedback: str = "Environment ready."
        self.done: bool = False
        self.step_count: int = 0
        self.max_steps: int = 20
        self.episode_reward: float = 0.0
        self.action_log: List[dict] = []

    def reset(self, task_id: str):
        self.task_id = task_id if task_id in TASKS else "task_easy"
        self.max_steps = MAX_STEPS.get(self.task_id, 20)

        raw = TASKS[self.task_id]
        self.tickets = []
        for t in raw:
            gt = TicketGroundTruth(
                id=t["id"],
                subject=t["subject"],
                body=t["body"],
                sender=t.get("sender", "customer"),
                sentiment=Sentiment(t.get("sentiment", "neutral")),
                created_at=t.get("created_at", "2026-04-03T00:00:00Z"),
                gt_category=TicketCategory(t["gt_category"]),
                gt_priority=TicketPriority(t["gt_priority"]),
                gt_needs_response=t.get("gt_needs_response", False),
                gt_needs_escalation=t.get("gt_needs_escalation", False),
                gt_response_keywords=t.get("gt_response_keywords", []),
            )
            self.tickets.append(gt)

        self.last_feedback = f"Environment reset. Task: {self.task_id}. You have {len(self.tickets)} tickets to process."
        self.done = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.action_log = []


env_state = EnvState()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _agent_ticket(gt: TicketGroundTruth) -> Ticket:
    """Convert ground-truth ticket to agent-visible ticket (no answers leaked)."""
    return Ticket(
        id=gt.id,
        subject=gt.subject,
        body=gt.body,
        sender=gt.sender,
        sentiment=gt.sentiment,
        status=gt.status,
        category=gt.category,
        priority=gt.priority,
        has_response=gt.has_response,
        response_text=gt.response_text,
        escalated=gt.escalated,
        notes=gt.notes,
        created_at=gt.created_at,
    )


def _build_observation() -> Observation:
    """Build the observation the agent sees."""
    agent_tickets = [_agent_ticket(t) for t in env_state.tickets]
    pending = sum(1 for t in env_state.tickets if t.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS))
    resolved = sum(1 for t in env_state.tickets if t.status == TicketStatus.CLOSED)
    escalated = sum(1 for t in env_state.tickets if t.escalated)

    return Observation(
        tickets=agent_tickets,
        last_feedback=env_state.last_feedback,
        current_score=round(_compute_grader_score(), 4),
        pending_count=pending,
        resolved_count=resolved,
        escalated_count=escalated,
        step_number=env_state.step_count,
        max_steps=env_state.max_steps,
    )


def _compute_grader_score() -> float:
    """
    Compute an overall grader score in [0.0, 1.0].

    Per ticket, we evaluate:
      - Category correct:  0.25 of ticket weight
      - Priority correct:  0.20 of ticket weight
      - Response handled:  0.25 of ticket weight
      - Escalation right:  0.20 of ticket weight
      - Proper closure:    0.10 of ticket weight

    Final score = average across all tickets.
    """
    if not env_state.tickets:
        return 0.0

    total = 0.0
    for t in env_state.tickets:
        ticket_score = 0.0
        # Only evaluate "absence of action" criteria once the agent has
        # interacted with the ticket (status != OPEN) or the episode is done.
        ticket_touched = t.status != TicketStatus.OPEN or env_state.done

        # Category (0.25)
        if t.category is not None:
            if t.category == t.gt_category.value:
                ticket_score += 0.25
            else:
                ticket_score += 0.0  # wrong category = no partial credit

        # Priority (0.20)
        if t.priority is not None:
            if t.priority == t.gt_priority.value:
                ticket_score += 0.20
            else:
                # Partial credit if within one level
                pri_order = ["low", "medium", "high", "critical"]
                if t.priority in pri_order:
                    diff = abs(pri_order.index(t.priority) - pri_order.index(t.gt_priority.value))
                    if diff == 1:
                        ticket_score += 0.10  # one off = half credit

        # Response (0.25)
        if t.gt_needs_response:
            if t.has_response and t.response_text and len(t.response_text) > 10:
                ticket_score += 0.15  # base credit for responding
                # Keyword bonus
                resp_lower = t.response_text.lower()
                matched = sum(1 for kw in t.gt_response_keywords if kw.lower() in resp_lower)
                if t.gt_response_keywords:
                    keyword_ratio = matched / len(t.gt_response_keywords)
                    ticket_score += 0.10 * keyword_ratio
        elif ticket_touched:
            # Should NOT have responded (e.g., security incident)
            # Only evaluate once the agent has started working on this ticket
            if not t.has_response:
                ticket_score += 0.25  # correct restraint
            else:
                ticket_score += 0.05  # responded unnecessarily, small penalty via less credit

        # Escalation (0.20)
        if t.gt_needs_escalation:
            if t.escalated:
                ticket_score += 0.20
        elif ticket_touched:
            # Only evaluate once the agent has started working on this ticket
            if not t.escalated:
                ticket_score += 0.20
            else:
                ticket_score += 0.0  # unnecessary escalation

        # Closure (0.10)
        if t.status == TicketStatus.CLOSED:
            # Only full credit if ticket was handled well
            if ticket_score >= 0.60:
                ticket_score += 0.10
            elif ticket_score >= 0.30:
                ticket_score += 0.05

        total += ticket_score

    return min(1.0, total / len(env_state.tickets))


def _process_action(action: Action) -> float:
    """Process one agent action. Returns the step reward."""
    atype = action.action_type.lower().strip()
    tid = action.ticket_id.strip()
    payload = action.payload.strip()

    # Handle noop
    if atype == "noop":
        env_state.last_feedback = "No operation performed. Consider taking a productive action."
        return -0.02

    # Find target ticket
    ticket = next((t for t in env_state.tickets if t.id == tid), None)
    if ticket is None:
        env_state.last_feedback = f"Error: Ticket '{tid}' not found. Available: {[t.id for t in env_state.tickets]}"
        return -0.05

    # ── CATEGORIZE ───────────────────────────────────────────────────────
    if atype == "categorize":
        if payload.lower() not in VALID_CATEGORIES:
            env_state.last_feedback = f"Invalid category '{payload}'. Valid: {sorted(VALID_CATEGORIES)}"
            return -0.05

        old_cat = ticket.category
        ticket.category = payload.lower()
        if ticket.status == TicketStatus.OPEN:
            ticket.status = TicketStatus.IN_PROGRESS

        if ticket.category == ticket.gt_category.value:
            env_state.last_feedback = f"Ticket {tid} categorized as '{payload}'."
            return 0.15 if old_cat != ticket.category else 0.0  # no double reward
        else:
            env_state.last_feedback = f"Ticket {tid} categorized as '{payload}'."
            return -0.05

    # ── SET PRIORITY ─────────────────────────────────────────────────────
    elif atype == "set_priority":
        if payload.lower() not in VALID_PRIORITIES:
            env_state.last_feedback = f"Invalid priority '{payload}'. Valid: {sorted(VALID_PRIORITIES)}"
            return -0.05

        old_pri = ticket.priority
        ticket.priority = payload.lower()
        if ticket.status == TicketStatus.OPEN:
            ticket.status = TicketStatus.IN_PROGRESS

        if ticket.priority == ticket.gt_priority.value:
            env_state.last_feedback = f"Ticket {tid} priority set to '{payload}'."
            return 0.10 if old_pri != ticket.priority else 0.0
        else:
            # Partial credit for being one off
            pri_order = ["low", "medium", "high", "critical"]
            if payload.lower() in pri_order:
                diff = abs(pri_order.index(payload.lower()) - pri_order.index(ticket.gt_priority.value))
                if diff == 1:
                    env_state.last_feedback = f"Ticket {tid} priority set to '{payload}' (close but not exact)."
                    return 0.03 if old_pri != ticket.priority else 0.0
            env_state.last_feedback = f"Ticket {tid} priority set to '{payload}'."
            return -0.05

    # ── RESPOND ──────────────────────────────────────────────────────────
    elif atype == "respond":
        if len(payload) < 10:
            env_state.last_feedback = "Response too short (minimum 10 characters). Please write a meaningful response."
            return -0.05

        ticket.has_response = True
        ticket.response_text = payload
        if ticket.status == TicketStatus.OPEN:
            ticket.status = TicketStatus.IN_PROGRESS

        if ticket.gt_needs_response:
            # Check keyword quality
            resp_lower = payload.lower()
            matched = sum(1 for kw in ticket.gt_response_keywords if kw.lower() in resp_lower)
            keyword_ratio = matched / max(1, len(ticket.gt_response_keywords))

            if keyword_ratio >= 0.5:
                env_state.last_feedback = f"Good response sent to {tid}. Relevant content detected."
                return 0.15
            else:
                env_state.last_feedback = f"Response sent to {tid}. Could be more specific/relevant."
                return 0.08
        else:
            env_state.last_feedback = f"Response sent to {tid}. Note: this ticket may not require a direct response."
            return -0.03

    # ── ESCALATE ─────────────────────────────────────────────────────────
    elif atype == "escalate":
        if ticket.escalated:
            env_state.last_feedback = f"Ticket {tid} was already escalated."
            return 0.0

        ticket.escalated = True
        ticket.status = TicketStatus.ESCALATED

        if ticket.gt_needs_escalation:
            env_state.last_feedback = f"Ticket {tid} escalated to senior team. Good judgment."
            return 0.20
        else:
            env_state.last_feedback = f"Ticket {tid} escalated. This ticket may not require escalation."
            return -0.10

    # ── ADD NOTE ─────────────────────────────────────────────────────────
    elif atype == "add_note":
        if len(payload) < 5:
            env_state.last_feedback = "Note too short. Please add a meaningful internal note."
            return -0.02

        ticket.notes.append(payload)
        if ticket.status == TicketStatus.OPEN:
            ticket.status = TicketStatus.IN_PROGRESS

        env_state.last_feedback = f"Note added to {tid}."
        return 0.03

    # ── CLOSE ────────────────────────────────────────────────────────────
    elif atype == "close":
        if ticket.status == TicketStatus.CLOSED:
            env_state.last_feedback = f"Ticket {tid} is already closed."
            return 0.0

        ticket.status = TicketStatus.CLOSED

        # Evaluate closure quality
        correct_cat = (ticket.category == ticket.gt_category.value) if ticket.category else False
        correct_pri = (ticket.priority == ticket.gt_priority.value) if ticket.priority else False
        correct_resp = ticket.has_response == ticket.gt_needs_response
        correct_esc = ticket.escalated == ticket.gt_needs_escalation

        checklist = [correct_cat, correct_pri, correct_resp, correct_esc]
        correct_count = sum(checklist)

        if correct_count == 4:
            env_state.last_feedback = f"Ticket {tid} closed. Excellent handling — all criteria met."
            return 0.15
        elif correct_count >= 3:
            env_state.last_feedback = f"Ticket {tid} closed. Good handling — {correct_count}/4 criteria met."
            return 0.08
        elif correct_count >= 2:
            env_state.last_feedback = f"Ticket {tid} closed. Partial handling — {correct_count}/4 criteria met."
            return 0.02
        else:
            missing = []
            if not correct_cat:
                missing.append("category")
            if not correct_pri:
                missing.append("priority")
            if not correct_resp:
                missing.append("response")
            if not correct_esc:
                missing.append("escalation")
            env_state.last_feedback = (
                f"Ticket {tid} closed prematurely. Issues: {', '.join(missing)}."
            )
            return -0.10

    else:
        env_state.last_feedback = f"Unknown action type: '{atype}'. Valid: categorize, set_priority, respond, escalate, add_note, close, noop"
        return -0.05


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "environment": "ticket-triage", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req else "task_easy"
    env_state.reset(task_id)
    obs = _build_observation()
    return ResetResponse(observation=obs, info={"task_id": env_state.task_id})


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if env_state.done:
        obs = _build_observation()
        return StepResponse(
            observation=obs,
            reward=0.0,
            done=True,
            info={"status": "episode_finished", "final_score": obs.current_score},
        )

    env_state.step_count += 1
    reward = _process_action(req.action)
    env_state.episode_reward += reward

    # Log the action
    env_state.action_log.append({
        "step": env_state.step_count,
        "action": req.action.model_dump(),
        "reward": reward,
    })

    # Check terminal conditions
    all_closed = all(t.status == TicketStatus.CLOSED for t in env_state.tickets)
    step_limit = env_state.step_count >= env_state.max_steps

    if all_closed:
        env_state.done = True
        env_state.last_feedback += " All tickets processed. Episode complete."
    elif step_limit:
        env_state.done = True
        env_state.last_feedback = f"Step limit ({env_state.max_steps}) reached. Episode ended."

    obs = _build_observation()
    return StepResponse(
        observation=obs,
        reward=round(reward, 4),
        done=env_state.done,
        info={
            "step": env_state.step_count,
            "episode_reward": round(env_state.episode_reward, 4),
            "final_score": obs.current_score if env_state.done else None,
        },
    )


@app.get("/state")
def get_state():
    score = _compute_grader_score()
    return State(
        task_id=env_state.task_id,
        step_count=env_state.step_count,
        max_steps=env_state.max_steps,
        score=round(score, 4),
        done=env_state.done,
        episode_reward=round(env_state.episode_reward, 4),
    ).model_dump()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()