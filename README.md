---
title: Ticket Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🎫 Ticket Triage — OpenEnv Environment

A production-grade OpenEnv environment that simulates **IT helpdesk ticket triage** — a task humans do millions of times daily. An AI agent must read customer support tickets, categorize them, assess priority, craft responses, decide on escalations, and properly close each case.

---

## 🌍 Why Ticket Triage?

| Dimension | Value |
|-----------|-------|
| **Real-world utility** | Customer support is a $400B+ industry. Every company needs ticket triage. |
| **Multi-step reasoning** | Agents must categorize → prioritize → investigate → respond → escalate → close |
| **Judgment required** | Security incidents need escalation, not responses. Angry customers need empathy. |
| **Natural difficulty** | Simple FAQ → billing disputes → security breaches with legal implications |
| **Evaluation clarity** | Each ticket has objective ground truth for grading |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────┐
│              Agent (inference.py)               │
│   OpenAI Client → JSON actions → HTTP POST     │
└──────────────────┬─────────────────────────────┘
                   │  POST /reset, /step, /state
                   ▼
┌────────────────────────────────────────────────┐
│          FastAPI Server (server/app.py)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  reset()  │  │  step()  │  │   state()    │ │
│  └──────────┘  └──────────┘  └──────────────┘ │
│         ┌──────────────────────────┐           │
│         │   Environment State      │           │
│         │   - Tickets (GT + Agent) │           │
│         │   - Scores & Rewards     │           │
│         │   - Step Counter         │           │
│         └──────────────────────────┘           │
└────────────────────────────────────────────────┘
```

---

## 🎮 Action Space

The agent can perform **7 action types**. Each action is a JSON object:

| Action | `action_type` | `payload` | Description |
|--------|---------------|-----------|-------------|
| **Categorize** | `categorize` | `technical\|billing\|shipping\|security\|general` | Set ticket category |
| **Set Priority** | `set_priority` | `low\|medium\|high\|critical` | Set ticket urgency |
| **Respond** | `respond` | Response text (≥10 chars) | Send customer-facing response |
| **Escalate** | `escalate` | (empty) | Send to senior team |
| **Add Note** | `add_note` | Note text (≥5 chars) | Add internal investigation note |
| **Close** | `close` | (empty) | Mark ticket as resolved |
| **No-op** | `noop` | (empty) | Skip turn (penalized) |

### Example Action
```json
{
  "action_type": "categorize",
  "ticket_id": "T1",
  "payload": "technical"
}
```

---

## 👁️ Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `tickets` | `List[Ticket]` | All tickets with current state |
| `last_feedback` | `str` | Feedback from the last action |
| `current_score` | `float` | Current grader score [0.0, 1.0] |
| `pending_count` | `int` | Tickets still open |
| `resolved_count` | `int` | Tickets closed |
| `escalated_count` | `int` | Tickets escalated |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum steps allowed |

### Ticket Fields
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Ticket ID (e.g., "T1") |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket description |
| `sender` | `str` | Customer email |
| `sentiment` | `str` | `positive\|neutral\|negative\|angry` |
| `status` | `str` | `open\|in_progress\|escalated\|closed` |
| `category` | `str?` | Agent-assigned category |
| `priority` | `str?` | Agent-assigned priority |
| `has_response` | `bool` | Whether agent has responded |
| `escalated` | `bool` | Whether ticket was escalated |
| `notes` | `List[str]` | Internal notes added by agent |

---

## 📋 Tasks

### Task Easy (2 tickets)
| Ticket | Subject | Expected Actions |
|--------|---------|-----------------|
| T1 | Login issue | categorize=technical, priority=medium, respond, close |
| T2 | Refund request | categorize=billing, priority=medium, respond, close |

**Difficulty**: Straightforward categorization and response. Both tickets have clear categories and need basic responses.

### Task Medium (3 tickets)
| Ticket | Subject | Expected Actions |
|--------|---------|-----------------|
| T3 | Password reset | categorize=technical, priority=low, respond, close |
| T4 | Missing order | categorize=shipping, priority=high, respond, close |
| T5 | Double charged | categorize=billing, priority=high, respond, **escalate**, close |

**Difficulty**: One ticket requires escalation. Agent must recognize that a double-charge (angry customer) needs senior team involvement.

### Task Hard (4 tickets)
| Ticket | Subject | Expected Actions |
|--------|---------|-----------------| 
| T6 | Account hacked | categorize=security, priority=critical, **escalate only** (no response!), close |
| T7 | API 500 error | categorize=technical, priority=critical, respond, **escalate**, close |
| T8 | Want to upgrade plan | categorize=billing, priority=low, respond, close |
| T9 | Legal threat - data breach | categorize=security, priority=critical, **escalate only** (no response!), close |

**Difficulty**: 
- T6 & T9 require **restraint** — security incidents should be escalated, NOT responded to (leaking information risk)
- T7 is a **production outage** requiring both response and escalation
- Four tickets with different priorities and handling strategies forces genuine triage

---

## 📊 Reward Function

### Step Rewards
| Action | Correct | Incorrect |
|--------|---------|-----------|
| Categorize | +0.15 | -0.05 |
| Set Priority | +0.10 | -0.05 |
| Respond (needed) | +0.08 to +0.15 | — |
| Respond (not needed) | -0.03 | — |
| Escalate (needed) | +0.20 | — |
| Escalate (not needed) | -0.10 | — |
| Add Note | +0.03 | -0.02 |
| Close (all correct) | +0.15 | — |
| Close (mostly correct) | +0.02 to +0.08 | — |
| Close (premature) | -0.10 | — |
| Noop | -0.02 | — |

### Final Grader Score (0.0 → 1.0)
Per ticket, evaluated on 5 criteria:
- **Category** (25%): Correct categorization
- **Priority** (20%): Correct priority (partial credit for ±1 level)
- **Response** (25%): Appropriate response with relevant keywords
- **Escalation** (20%): Correct escalation decision
- **Closure** (10%): Proper closure after handling

Score = average across all tickets, capped at 1.0.

---

## 🚀 Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py
```

### Docker

```bash
# Build
docker build -t ticket-triage .

# Run
docker run -p 8000:8000 ticket-triage

# Test health
curl http://localhost:8000/health
# {"status":"ok"}

# Test reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

### Hugging Face Spaces

This environment is deployed as a Hugging Face Space with Docker:

```bash
# The Dockerfile handles everything — just deploy
# Tag your space with "openenv"
```

---

## 📈 Baseline Scores

| Task | Expected Score Range | Model |
|------|---------------------|-------|
| task_easy | 0.75 – 0.95 | gpt-4o-mini |
| task_medium | 0.55 – 0.80 | gpt-4o-mini |
| task_hard | 0.35 – 0.65 | gpt-4o-mini |

*Scores vary by model capability. Frontier models (GPT-4o, Claude) should score higher.*

---

## 🔧 API Reference

### `POST /reset`
Reset the environment to start a new episode.

**Request:**
```json
{"task_id": "task_easy"}
```

**Response:**
```json
{
  "observation": {
    "tickets": [...],
    "last_feedback": "Environment reset. Task: task_easy...",
    "current_score": 0.0,
    "pending_count": 2,
    "resolved_count": 0,
    "escalated_count": 0,
    "step_number": 0,
    "max_steps": 20
  },
  "info": {"task_id": "task_easy"}
}
```

### `POST /step`
Take an action in the environment.

**Request:**
```json
{
  "action": {
    "action_type": "categorize",
    "ticket_id": "T1",
    "payload": "technical"
  }
}
```

**Response:**
```json
{
  "observation": {...},
  "reward": 0.15,
  "done": false,
  "info": {"step": 1, "episode_reward": 0.15}
}
```

### `GET /state`
Get the current environment state.

**Response:**
```json
{
  "task_id": "task_easy",
  "step_count": 3,
  "max_steps": 20,
  "score": 0.45,
  "done": false,
  "episode_reward": 0.35
}
```

### `GET /health`
Health check endpoint.

---

## 📁 Project Structure

```
openenv-support-triage/
├── server/
│   ├── __init__.py         # Package init
│   ├── models.py           # Pydantic models (Action, Observation, State, Ticket)
│   └── app.py              # FastAPI server with environment logic
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

---

## 📄 License

MIT
