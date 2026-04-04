"""
Ticket Triage Environment Implementation.

Subclasses the OpenEnv Environment base class to provide
step/reset/state for IT helpdesk ticket triage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Optional, List, Dict
from uuid import uuid4
from enum import Enum

from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.interfaces import Environment
from models import TriageObservation


# ─── Enums ───────────────────────────────────────────────────────────────────

class Category(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    SHIPPING = "shipping"
    SECURITY = "security"
    GENERAL = "general"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── Ticket data ─────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "task_easy": {
        "max_steps": 20,
        "tickets": [
            {
                "id": "T1", "subject": "Can't login to my account",
                "body": "Hi, I've been trying to login but keep getting 'invalid credentials'. I reset my password yesterday. Username: john.doe@email.com",
                "sender": "john.doe@email.com", "sentiment": "frustrated",
            },
            {
                "id": "T2", "subject": "Damaged product received",
                "body": "Order #9921 arrived with a cracked screen. I paid $299 and the box was clearly mishandled. I want a refund.",
                "sender": "sarah.m@email.com", "sentiment": "angry",
            },
        ],
        "ground_truth": {
            "T1": {"category": "technical", "priority": "medium", "needs_response": True, "needs_escalation": False, "response_keywords": ["password", "reset", "login", "credentials"]},
            "T2": {"category": "billing", "priority": "medium", "needs_response": True, "needs_escalation": False, "response_keywords": ["refund", "damaged", "replacement", "order"]},
        },
    },
    "task_medium": {
        "max_steps": 25,
        "tickets": [
            {
                "id": "T3", "subject": "How to reset my password?",
                "body": "I forgot my password and can't find the reset option. Can someone help?",
                "sender": "mike.j@email.com", "sentiment": "neutral",
            },
            {
                "id": "T4", "subject": "Package never arrived",
                "body": "Order #443 was supposed to arrive 5 days ago but tracking shows it's stuck. This is unacceptable! I need this for a presentation tomorrow.",
                "sender": "emma.w@email.com", "sentiment": "angry",
            },
            {
                "id": "T5", "subject": "Charged twice for subscription",
                "body": "My credit card was charged $49.99 TWICE for my monthly subscription. Order refs: #SUB-7781, #SUB-7782. Fix this immediately or I'm calling my bank.",
                "sender": "alex.r@email.com", "sentiment": "angry",
            },
        ],
        "ground_truth": {
            "T3": {"category": "technical", "priority": "low", "needs_response": True, "needs_escalation": False, "response_keywords": ["password", "reset", "link", "email"]},
            "T4": {"category": "shipping", "priority": "high", "needs_response": True, "needs_escalation": False, "response_keywords": ["tracking", "delivery", "shipping", "order"]},
            "T5": {"category": "billing", "priority": "high", "needs_response": True, "needs_escalation": True, "response_keywords": ["charge", "refund", "billing", "subscription"]},
        },
    },
    "task_hard": {
        "max_steps": 30,
        "tickets": [
            {
                "id": "T6", "subject": "URGENT: Account hacked!",
                "body": "Someone changed my password and email on my account! I can see purchases I didn't make totaling $2,847. My account ID is #ACC-9912. HELP!",
                "sender": "victim@email.com", "sentiment": "panicked",
            },
            {
                "id": "T7", "subject": "API returning 500 errors",
                "body": "Our production system has been getting 500 errors from your API v2 endpoint since 3am. This is affecting 50,000+ users. We need immediate resolution.",
                "sender": "cto@enterprise.com", "sentiment": "urgent",
            },
            {
                "id": "T8", "subject": "Want to upgrade my plan",
                "body": "I'm on the Basic plan and want to switch to Enterprise. Can you tell me about the features and pricing?",
                "sender": "prospect@company.com", "sentiment": "positive",
            },
            {
                "id": "T9", "subject": "Legal threat - data breach",
                "body": "I received an email saying my data was part of a breach. Unless I receive a full report of what was compromised within 48 hours, my lawyer will be in touch.",
                "sender": "angry.customer@email.com", "sentiment": "threatening",
            },
        ],
        "ground_truth": {
            "T6": {"category": "security", "priority": "critical", "needs_response": False, "needs_escalation": True, "response_keywords": []},
            "T7": {"category": "technical", "priority": "critical", "needs_response": True, "needs_escalation": True, "response_keywords": ["api", "error", "investigating", "priority"]},
            "T8": {"category": "billing", "priority": "low", "needs_response": True, "needs_escalation": False, "response_keywords": ["upgrade", "plan", "enterprise", "features"]},
            "T9": {"category": "security", "priority": "critical", "needs_response": False, "needs_escalation": True, "response_keywords": []},
        },
    },
}


# ─── Environment ─────────────────────────────────────────────────────────────

class TriageEnvironment(Environment):
    """
    IT Helpdesk Ticket Triage Environment.

    Agents process support tickets by categorizing, prioritizing,
    responding, escalating, and closing them. Scored 0-1 by a
    deterministic grader across 5 criteria per ticket.
    """

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._tickets: Dict[str, Dict[str, Any]] = {}
        self._ground_truth: Dict[str, Dict[str, Any]] = {}
        self._max_steps = 20
        self._task_id = "task_easy"

    # ── reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        tid = task_id or kwargs.get("task_id", "task_easy")
        if tid not in TASKS:
            tid = "task_easy"
        self._task_id = tid
        task = TASKS[tid]
        self._max_steps = task["max_steps"]
        self._ground_truth = task["ground_truth"]
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._tickets = {}
        for t in task["tickets"]:
            self._tickets[t["id"]] = {
                **t,
                "status": "open",
                "category": None,
                "priority": None,
                "has_response": False,
                "response_text": "",
                "escalated": False,
                "notes": [],
            }

        obs_data = self._build_obs_meta("Environment reset. Begin triaging tickets.")
        return TriageObservation(
            done=False,
            reward=0.0,
            **obs_data,
        )

    # ── step ─────────────────────────────────────────────────────────────

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        step = self._state.step_count

        # Parse action fields from the generic Action
        act_data = action.model_dump() if hasattr(action, "model_dump") else {}
        action_type = act_data.get("action_type", "noop")
        ticket_id = act_data.get("ticket_id", "")
        payload = act_data.get("payload", "")

        reward = 0.0
        feedback = ""

        # ── Validate ─────────────────────────────────────────────────
        if action_type == "noop" or not action_type:
            reward = -0.02
            feedback = "No operation performed. Consider taking a productive action."
            return self._make_obs(reward, feedback, step)

        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            reward = -0.05
            feedback = f"Invalid ticket_id: {ticket_id}."
            return self._make_obs(reward, feedback, step)

        if ticket["status"] == "closed" and action_type != "add_note":
            reward = -0.05
            feedback = f"Ticket {ticket_id} is already closed."
            return self._make_obs(reward, feedback, step)

        gt = self._ground_truth.get(ticket_id, {})

        # ── Execute action ───────────────────────────────────────────
        if action_type == "categorize":
            cat = payload.lower().strip()
            valid = [c.value for c in Category]
            if cat not in valid:
                reward = -0.05
                feedback = f"Invalid category '{cat}'. Use one of: {valid}"
            else:
                ticket["category"] = cat
                reward = 0.15 if cat == gt.get("category") else 0.03
                feedback = f"Ticket {ticket_id} categorized as '{cat}'."
                if cat != gt.get("category"):
                    feedback += " (close but not exact)."

        elif action_type == "set_priority":
            pri = payload.lower().strip()
            valid = [p.value for p in Priority]
            if pri not in valid:
                reward = -0.05
                feedback = f"Invalid priority '{pri}'. Use one of: {valid}"
            else:
                ticket["priority"] = pri
                reward = 0.10 if pri == gt.get("priority") else 0.03
                feedback = f"Ticket {ticket_id} priority set to '{pri}'."
                if pri != gt.get("priority"):
                    feedback += " (close but not exact)."

        elif action_type == "respond":
            if gt.get("needs_response") is False:
                reward = -0.15
                feedback = f"Ticket {ticket_id} should NOT receive a direct response (security/sensitive)."
            elif len(payload) < 10:
                reward = -0.05
                feedback = "Response too short. Provide a substantive reply."
            else:
                ticket["has_response"] = True
                ticket["response_text"] = payload
                kws = gt.get("response_keywords", [])
                hits = sum(1 for k in kws if k.lower() in payload.lower())
                if hits >= max(1, len(kws) // 2):
                    reward = 0.15
                    feedback = f"Good response sent to {ticket_id}. Relevant content detected."
                else:
                    reward = 0.05
                    feedback = f"Response sent to {ticket_id} but could be more relevant."

        elif action_type == "escalate":
            if gt.get("needs_escalation"):
                ticket["escalated"] = True
                reward = 0.20
                feedback = f"Ticket {ticket_id} escalated to senior team. Good judgment."
            else:
                ticket["escalated"] = True
                reward = -0.10
                feedback = f"Ticket {ticket_id} escalated. This ticket may not require escalation."

        elif action_type == "add_note":
            ticket["notes"].append(payload)
            reward = 0.02
            feedback = f"Internal note added to {ticket_id}."

        elif action_type == "close":
            ticket["status"] = "closed"
            score = self._grade_ticket(ticket_id)
            if score >= 0.9:
                reward = 0.15
                feedback = f"Ticket {ticket_id} closed. Excellent handling — all criteria met."
            elif score >= 0.6:
                met = int(score * 4)
                reward = 0.08
                feedback = f"Ticket {ticket_id} closed. Good handling — {met}/4 criteria met."
            elif score >= 0.3:
                met = int(score * 4)
                reward = 0.02
                feedback = f"Ticket {ticket_id} closed. Partial handling — {met}/4 criteria met."
            else:
                reward = -0.10
                feedback = f"Ticket {ticket_id} closed prematurely — most criteria not met."
        else:
            reward = -0.05
            feedback = f"Unknown action_type: {action_type}"

        return self._make_obs(reward, feedback, step)

    # ── state property ───────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ── helpers ───────────────────────────────────────────────────────────

    def _make_obs(self, reward: float, feedback: str, step: int) -> TriageObservation:
        all_closed = all(t["status"] == "closed" for t in self._tickets.values())
        done = all_closed or step >= self._max_steps

        if all_closed and not done:
            feedback += " All tickets processed. Episode complete."
            done = True
        elif step >= self._max_steps:
            feedback = f"Step limit ({self._max_steps}) reached. Episode ended."
            done = True

        score = self._compute_score()
        obs_data = self._build_obs_meta(feedback, score)

        return TriageObservation(
            done=done,
            reward=reward,
            **obs_data,
        )

    def _build_obs_meta(self, feedback: str, score: float = 0.0) -> Dict[str, Any]:
        ticket_list = []
        pending = resolved = escalated = 0
        for t in self._tickets.values():
            ticket_list.append({
                "id": t["id"],
                "subject": t["subject"],
                "body": t["body"],
                "sender": t.get("sender", ""),
                "sentiment": t.get("sentiment", "neutral"),
                "status": t["status"],
                "category": t["category"],
                "priority": t["priority"],
                "has_response": t["has_response"],
                "escalated": t["escalated"],
            })
            if t["status"] == "open":
                pending += 1
            else:
                resolved += 1
            if t["escalated"]:
                escalated += 1

        return {
            "tickets": ticket_list,
            "last_feedback": feedback,
            "pending_count": pending,
            "resolved_count": resolved,
            "escalated_count": escalated,
            "step_number": self._state.step_count,
            "max_steps": self._max_steps,
            "current_score": round(score, 4),
        }

    def _grade_ticket(self, tid: str) -> float:
        t = self._tickets[tid]
        gt = self._ground_truth.get(tid, {})
        
        score = 0.0
        
        # Category (25%)
        if gt.get("category"):
            if t["category"] == gt["category"]:
                score += 0.25
                
        # Priority (20% with partial credit)
        if gt.get("priority") and t.get("priority"):
            p_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            t_p = p_levels.get(t["priority"], -1)
            gt_p = p_levels.get(gt["priority"], -1)
            if t_p != -1 and gt_p != -1:
                diff = abs(t_p - gt_p)
                if diff == 0:
                    score += 0.20
                elif diff == 1:
                    score += 0.10
                    
        # Response (25%)
        if gt.get("needs_response"):
            if t["has_response"]:
                score += 0.25
        else:
            if not t["has_response"]:
                score += 0.25
                
        # Escalation (20%)
        if gt.get("needs_escalation"):
            if t["escalated"]:
                score += 0.20
        else:
            if not t["escalated"]:
                score += 0.20
                
        # Closure (10%)
        if t["status"] == "closed":
            score += 0.10
            
        return score

    def _compute_score(self) -> float:
        if not self._tickets:
            return 0.0
        total = sum(self._grade_ticket(tid) for tid in self._tickets)
        return total / len(self._tickets)
