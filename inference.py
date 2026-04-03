"""
Enhanced inference script for Ticket Triage OpenEnv.

Reads credentials from environment variables:
    API_BASE_URL       — LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME         — Model identifier (default: gpt-4o-mini)
    HF_TOKEN           — API key for the LLM (required, no default)
    LOCAL_IMAGE_NAME   — Optional, for from_docker_image() usage

Usage:
    python inference.py                     # Run all tasks
    python inference.py --task task_easy     # Run a single task
    python inference.py --task task_hard --verbose  # Verbose output

Emits structured stdout logs in [START], [STEP], [END] format.
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME (not HF_TOKEN)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Enhanced System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert customer support triage agent. Your job is to process support tickets by performing the right actions in the right order.

## Available Actions (respond with exactly ONE JSON object per turn)

1. **Categorize** a ticket:
   {"action_type": "categorize", "ticket_id": "T1", "payload": "<category>"}
   Categories: technical, billing, shipping, security, general

2. **Set priority** for a ticket:
   {"action_type": "set_priority", "ticket_id": "T1", "payload": "<priority>"}
   Priorities: low, medium, high, critical

3. **Respond** to a ticket (write a helpful customer-facing response):
   {"action_type": "respond", "ticket_id": "T1", "payload": "Your detailed response here..."}

4. **Escalate** a ticket to senior team:
   {"action_type": "escalate", "ticket_id": "T1", "payload": ""}

5. **Add internal note** to a ticket:
   {"action_type": "add_note", "ticket_id": "T1", "payload": "Your internal note here..."}

6. **Close** a ticket (only after properly handling it):
   {"action_type": "close", "ticket_id": "T1", "payload": ""}

## Ticket Processing Strategy — FOLLOW THIS ORDER

Process tickets ONE AT A TIME. For each ticket, follow this exact sequence:
1. **Categorize** the ticket first
2. **Set priority** based on urgency and impact
3. **Respond** to the customer (if appropriate — see exceptions below)
4. **Escalate** if the situation warrants it (see guidelines below)
5. **Close** the ticket after all necessary actions are taken
Then move to the next ticket.

## CRITICAL Rules

### Security Incidents (category = security)
- **NEVER** respond to security tickets. Responding may leak sensitive information.
- **ALWAYS** escalate security tickets immediately after categorizing and setting priority.
- Set priority to "critical" for security incidents.
- Close the ticket after escalation (do NOT respond).

### Escalation Guidelines
- **Always escalate**: Security incidents, billing disputes (especially double charges), complex technical issues blocking production
- **Consider escalating**: Angry customers threatening legal action, issues requiring refunds over $100
- Escalation is for cases that need senior team review — don't escalate simple questions

### Response Guidelines
- Write substantive responses (minimum 10 characters, aim for 50+)
- **Address the specific issue** mentioned in the ticket (use relevant details like order numbers, error codes, etc.)
- For angry customers: Acknowledge their frustration, apologize sincerely, explain what you'll do to fix it
- Include relevant keywords from the ticket (e.g., "password reset", "refund", "tracking", "API error")
- Be professional, empathetic, and solution-oriented

### Priority Guidelines
- **critical**: Security breaches, system-wide outages, data loss
- **high**: Production-blocking issues, billing errors, urgent orders, angry customers with significant problems
- **medium**: Standard login/account issues, routine refunds, general technical questions
- **low**: FAQ-type questions, feature requests, information queries

## Response Format
Return ONLY a single valid JSON object. No extra text, no markdown, no explanation.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_json_safely(text: str) -> dict:
    """Try to parse JSON from LLM output, handling common issues."""
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def run_task(task_id: str, verbose: bool = False) -> float:
    """Run one task and return the final grader score."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # ── Reset environment ────────────────────────────────────────────────
    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
        return 0.0

    obs = reset_data["observation"]

    # ── [START] log ──────────────────────────────────────────────────────
    print(
        json.dumps(
            {
                "event": "[START]",
                "task_id": task_id,
                "timestamp": now_iso(),
                "ticket_count": len(obs["tickets"]),
                "max_steps": obs.get("max_steps", 20),
            }
        )
    )
    sys.stdout.flush()

    done = False
    step_num = 0
    total_reward = 0.0
    history = []
    max_retries = 5

    # ── Agent loop ───────────────────────────────────────────────────────
    while not done and step_num < 40:
        step_num += 1

        # Build the user prompt with current observation
        ticket_summaries = []
        for t in obs["tickets"]:
            summary = {
                "id": t["id"],
                "subject": t["subject"],
                "body": t["body"][:500],  # Truncate very long bodies
                "sender": t.get("sender", "unknown"),
                "sentiment": t.get("sentiment", "neutral"),
                "status": t.get("status", "open"),
                "category": t.get("category"),
                "priority": t.get("priority"),
                "has_response": t.get("has_response", False),
                "escalated": t.get("escalated", False),
            }
            ticket_summaries.append(summary)

        prompt = (
            f"Current tickets:\n{json.dumps(ticket_summaries, indent=2)}\n\n"
            f"Feedback from last action: {obs.get('last_feedback', 'None')}\n"
            f"Current score: {obs.get('current_score', 0.0)}\n"
            f"Pending: {obs.get('pending_count', '?')} | "
            f"Resolved: {obs.get('resolved_count', '?')} | "
            f"Escalated: {obs.get('escalated_count', '?')}\n"
            f"Step: {obs.get('step_number', step_num)}/{obs.get('max_steps', 20)}\n\n"
            f"Process the tickets one at a time following the strategy. "
            f"What is your next action? Return a single JSON object."
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": prompt}]
        )

        # Call LLM with retry logic
        action = None
        for attempt in range(max_retries):
            try:
                llm_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.05,  # Low temperature for consistent behavior
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                action_text = llm_resp.choices[0].message.content.strip()
                action = parse_json_safely(action_text)

                if action is not None:
                    # Ensure all required fields
                    action.setdefault("action_type", "noop")
                    action.setdefault("ticket_id", "")
                    action.setdefault("payload", "")

                    # Keep conversation history
                    history.append({"role": "user", "content": prompt})
                    history.append({"role": "assistant", "content": action_text})

                    # Keep history manageable (last 12 turns = 24 messages)
                    if len(history) > 24:
                        history = history[-24:]

                    break
                else:
                    if verbose:
                        print(
                            f"  [RETRY {attempt+1}] Could not parse JSON: {action_text[:100]}",
                            file=sys.stderr,
                        )

            except Exception as e:
                wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                print(
                    f"  [RETRY {attempt+1}/{max_retries}] LLM error, waiting {wait_time}s: {str(e)[:80]}",
                    file=sys.stderr,
                )
                time.sleep(wait_time)

        if action is None:
            print(f"[ERROR] All retries failed at step {step_num}", file=sys.stderr)
            action = {"action_type": "noop", "ticket_id": "", "payload": ""}

        # ── Step environment ─────────────────────────────────────────────
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            break

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        total_reward += reward

        # ── [STEP] log ───────────────────────────────────────────────────
        step_log = {
            "event": "[STEP]",
            "task_id": task_id,
            "step": step_num,
            "action": {
                "action_type": action.get("action_type", "noop"),
                "ticket_id": action.get("ticket_id", ""),
                "payload": str(action.get("payload", ""))[:100],
            },
            "reward": round(reward, 4),
            "total_reward": round(total_reward, 4),
            "done": done,
            "feedback": obs.get("last_feedback", ""),
            "current_score": obs.get("current_score", 0.0),
            "timestamp": now_iso(),
        }
        print(json.dumps(step_log))
        sys.stdout.flush()

        if verbose:
            print(
                f"  Step {step_num}: {action.get('action_type')} "
                f"T={action.get('ticket_id')} → reward={reward:+.2f} "
                f"(total={total_reward:.2f}, score={obs.get('current_score', 0):.4f})",
                file=sys.stderr,
            )

        # Rate-limit to avoid API throttling
        time.sleep(2.5)  # Respect rate limits (Groq free: ~30 req/min)

    # ── Get final score ──────────────────────────────────────────────────
    final_score = obs.get("current_score", 0.0)

    # ── [END] log ────────────────────────────────────────────────────────
    print(
        json.dumps(
            {
                "event": "[END]",
                "task_id": task_id,
                "steps": step_num,
                "total_reward": round(total_reward, 4),
                "final_score": round(final_score, 4),
                "timestamp": now_iso(),
            }
        )
    )
    sys.stdout.flush()

    return final_score


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket Triage Inference Agent")
    parser.add_argument(
        "--task",
        choices=["task_easy", "task_medium", "task_hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose step output"
    )
    args = parser.parse_args()

    print(f"=== Ticket Triage Enhanced Inference ===", file=sys.stderr)
    print(f"API: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Env: {ENV_URL}", file=sys.stderr)
    print(f"========================================", file=sys.stderr)

    if not HF_TOKEN:
        print(
            "WARNING: HF_TOKEN not set. Please set HF_TOKEN environment variable.",
            file=sys.stderr,
        )

    tasks = (
        [args.task]
        if args.task != "all"
        else ["task_easy", "task_medium", "task_hard"]
    )

    scores = {}
    for task in tasks:
        print(f"\n--- Starting {task} ---", file=sys.stderr)
        score = run_task(task, verbose=args.verbose)
        scores[task] = score
        print(f"--- {task}: score = {score:.4f} ---\n", file=sys.stderr)
        time.sleep(1)

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"  Final Scores", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)
    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:15s} {bar} {score:.4f}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'Average':15s} {'':20s} {avg:.4f}", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)
