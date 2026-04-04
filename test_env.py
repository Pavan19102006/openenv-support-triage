#!/usr/bin/env python3
"""
Deterministic test suite for the Ticket Triage OpenEnv environment.
Tests the server endpoints, reward logic, grader scoring, and edge cases
WITHOUT needing any LLM API key.
"""

import sys
import json
import requests

BASE = "http://localhost:7860"
PASS = 0
FAIL = 0
TOTAL = 0


def log(msg, ok=True):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if ok:
        PASS += 1
        print(f"  ✅ {msg}")
    else:
        FAIL += 1
        print(f"  ❌ {msg}")


def reset(task_id):
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def step(action_type, ticket_id="", payload=""):
    r = requests.post(f"{BASE}/step", json={
        "action": {"action_type": action_type, "ticket_id": ticket_id, "payload": payload}
    }, timeout=10)
    r.raise_for_status()
    return r.json()


def state():
    r = requests.get(f"{BASE}/state", timeout=10)
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Health & Root Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_endpoints():
    print("\n🔬 Test 1: Health & Root Endpoints")

    r = requests.get(f"{BASE}/health")
    log(f"GET /health → {r.status_code}", r.status_code == 200 and r.json()["status"] == "healthy")

    r = requests.get(f"{BASE}/")
    data = r.json()
    log(f"GET / → {data.get('environment')}", data.get("environment") == "ticket-triage")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Reset Endpoint (OpenEnv format: observation + reward + done)
# ═══════════════════════════════════════════════════════════════════════════════

def test_reset():
    print("\n🔬 Test 2: Reset Endpoint")

    for task_id, expected_count in [("task_easy", 2), ("task_medium", 3), ("task_hard", 4)]:
        data = reset(task_id)
        obs = data["observation"]

        # Verify OpenEnv response format
        has_reward = "reward" in data
        has_done = "done" in data
        log(
            f"Reset {task_id}: {len(obs['tickets'])} tickets, step={obs['step_number']}, score={obs['current_score']}, format_ok={has_reward and has_done}",
            len(obs["tickets"]) == expected_count
            and obs["step_number"] == 0
            and obs["current_score"] == 0.0
            and obs["pending_count"] == expected_count
            and has_reward
            and has_done
        )

    # Invalid task falls back to task_easy
    data = reset("invalid_task")
    log(f"Invalid task fallback: {len(data['observation']['tickets'])} tickets", len(data["observation"]["tickets"]) == 2)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Perfect Run — task_easy
# ═══════════════════════════════════════════════════════════════════════════════

def test_perfect_easy():
    print("\n🔬 Test 3: Perfect Run — task_easy")
    reset("task_easy")

    total_reward = 0.0

    # T1: technical, medium, respond, close
    r = step("categorize", "T1", "technical")
    log(f"T1 categorize=technical → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("set_priority", "T1", "medium")
    log(f"T1 priority=medium → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("respond", "T1", "We're sorry about the login issue. Please try resetting your password using the 'Forgot Password' link on the login page. If you still cannot access your account, we can reset it for you.")
    log(f"T1 respond → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("close", "T1", "")
    log(f"T1 close → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    # T2: billing, medium, respond, close
    r = step("categorize", "T2", "billing")
    log(f"T2 categorize=billing → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("set_priority", "T2", "medium")
    log(f"T2 priority=medium → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("respond", "T2", "We apologize for the damage to your order #9921. We will process a full refund immediately. No need to return the damaged item.")
    log(f"T2 respond → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]

    r = step("close", "T2", "")
    log(f"T2 close → reward={r['reward']}", r["reward"] > 0)
    total_reward += r["reward"]
    log(f"T2 close done={r['done']}", r["done"] is True)

    final_score = r["observation"]["current_score"]
    log(f"Final score: {final_score} (expected ≥ 0.90)", final_score >= 0.90)
    log(f"Total reward: {round(total_reward, 4)}", total_reward > 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Perfect Run — task_medium (with escalation on T5)
# ═══════════════════════════════════════════════════════════════════════════════

def test_perfect_medium():
    print("\n🔬 Test 4: Perfect Run — task_medium")
    reset("task_medium")

    total_reward = 0.0

    # T3: technical, low, respond, close
    for atype, tid, payload, label in [
        ("categorize", "T3", "technical", "T3 cat"),
        ("set_priority", "T3", "low", "T3 pri"),
        ("respond", "T3", "To reset your password, go to the login page and click the 'Forgot Password' link. Enter your email and follow the reset instructions.", "T3 resp"),
        ("close", "T3", "", "T3 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    # T4: shipping, high, respond, close
    for atype, tid, payload, label in [
        ("categorize", "T4", "shipping", "T4 cat"),
        ("set_priority", "T4", "high", "T4 pri"),
        ("respond", "T4", "We apologize about your order #443. We're tracking delivery with tracking number TRACK-8812 and will escalate with the carrier to ensure fast delivery.", "T4 resp"),
        ("close", "T4", "", "T4 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    # T5: billing, high, respond, ESCALATE, close
    for atype, tid, payload, label in [
        ("categorize", "T5", "billing", "T5 cat"),
        ("set_priority", "T5", "high", "T5 pri"),
        ("respond", "T5", "We sincerely apologize for the duplicate charge. We will reverse the extra charge and issue a refund for transaction TXN-001123 immediately.", "T5 resp"),
        ("escalate", "T5", "", "T5 escalate"),
        ("close", "T5", "", "T5 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    log(f"Episode done={r['done']}", r["done"] is True)
    final_score = r["observation"]["current_score"]
    log(f"Final score: {final_score} (expected ≥ 0.85)", final_score >= 0.85)
    log(f"Total reward: {round(total_reward, 4)}", total_reward > 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Perfect Run — task_hard (security no-response, angry empathy)
# ═══════════════════════════════════════════════════════════════════════════════

def test_perfect_hard():
    print("\n🔬 Test 5: Perfect Run — task_hard")
    reset("task_hard")

    total_reward = 0.0

    # T6: security, critical, escalate ONLY (no respond!), then close
    for atype, tid, payload, label in [
        ("categorize", "T6", "security", "T6 cat"),
        ("set_priority", "T6", "critical", "T6 pri"),
        ("escalate", "T6", "", "T6 escalate"),
        ("close", "T6", "", "T6 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    # T7: technical, critical, respond, escalate, close
    for atype, tid, payload, label in [
        ("categorize", "T7", "technical", "T7 cat"),
        ("set_priority", "T7", "critical", "T7 pri"),
        ("respond", "T7", "We're aware of the API v2 error and our engineering team is actively investigating the 500 errors on the /api/v2/orders endpoint. This is our top priority and we'll provide an update shortly.", "T7 resp"),
        ("escalate", "T7", "", "T7 escalate"),
        ("close", "T7", "", "T7 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    # T8: billing, low, respond (upgrade inquiry), close — NO escalation
    for atype, tid, payload, label in [
        ("categorize", "T8", "billing", "T8 cat"),
        ("set_priority", "T8", "low", "T8 pri"),
        ("respond", "T8", "Thank you for your interest in upgrading your plan! Our Enterprise plan includes advanced features like priority support, custom integrations, and dedicated account management. I'd be happy to help you with the upgrade process.", "T8 resp"),
        ("close", "T8", "", "T8 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    # T9: security, critical, escalate ONLY (legal data breach — no respond!), close
    for atype, tid, payload, label in [
        ("categorize", "T9", "security", "T9 cat"),
        ("set_priority", "T9", "critical", "T9 pri"),
        ("escalate", "T9", "", "T9 escalate"),
        ("close", "T9", "", "T9 close"),
    ]:
        r = step(atype, tid, payload)
        total_reward += r["reward"]
        log(f"{label} → reward={r['reward']}", r["reward"] >= 0)

    log(f"Episode done={r['done']}", r["done"] is True)
    final_score = r["observation"]["current_score"]
    log(f"Final score: {final_score} (expected ≥ 0.80)", final_score >= 0.80)
    log(f"Total reward: {round(total_reward, 4)}", total_reward > 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n🔬 Test 6: Edge Cases")
    reset("task_easy")

    # Invalid ticket ID
    r = step("categorize", "TX99", "technical")
    log(f"Invalid ticket ID → reward={r['reward']}", r["reward"] < 0)

    # Invalid category
    r = step("categorize", "T1", "unknown_category")
    log(f"Invalid category → reward={r['reward']}", r["reward"] < 0)

    # Invalid priority
    r = step("set_priority", "T1", "extreme")
    log(f"Invalid priority → reward={r['reward']}", r["reward"] < 0)

    # Response too short
    r = step("respond", "T1", "ok")
    log(f"Short response → reward={r['reward']}", r["reward"] < 0)

    # Note (any length is accepted)
    r = step("add_note", "T1", "hi")
    log(f"Short note → reward={r['reward']}", r["reward"] >= 0)

    # Noop penalty
    r = step("noop", "", "")
    log(f"Noop → reward={r['reward']}", r["reward"] < 0)

    # Unknown action
    r = step("fly_ticket", "T1", "")
    log(f"Unknown action → reward={r['reward']}", r["reward"] < 0)

    # Wrong category gives partial credit (not negative)
    r = step("categorize", "T1", "shipping")
    log(f"Wrong category → reward={r['reward']}", r["reward"] > 0)

    # Now fix and close properly
    r = step("categorize", "T1", "technical")
    log(f"Correct re-categorize (no double reward) → reward={r['reward']}", r["reward"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: State Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def test_state_endpoint():
    print("\n🔬 Test 7: State Endpoint")
    reset("task_easy")
    s = state()
    log(f"State: task={s['task_id']}, step={s['step_count']}, done={s['done']}",
        s["task_id"] == "task_easy" and s["step_count"] == 0 and s["done"] is False)

    step("categorize", "T1", "technical")
    s = state()
    log(f"After action: step={s['step_count']}, score={s['score']}", s["step_count"] == 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Episode termination at step limit
# ═══════════════════════════════════════════════════════════════════════════════

def test_step_limit():
    print("\n🔬 Test 8: Step Limit Termination")
    reset("task_easy")  # max_steps = 20

    for i in range(21):
        r = step("noop", "", "")
        if r["done"]:
            break

    log(f"Episode ended at step limit: done={r['done']}", r["done"] is True)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Double close / actions on closed ticket
# ═══════════════════════════════════════════════════════════════════════════════

def test_double_close():
    print("\n🔬 Test 9: Double Close & Actions After Close")
    reset("task_easy")

    step("categorize", "T1", "technical")
    step("set_priority", "T1", "medium")
    step("respond", "T1", "Please reset your password via the forgot password link on your account page.")
    step("close", "T1", "")

    r = step("close", "T1", "")
    log(f"Double close → reward={r['reward']}", r["reward"] < 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: OpenEnv Response Format Validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_openenv_response_format():
    print("\n🔬 Test 10: OpenEnv Response Format")

    # Test reset response format
    data = reset("task_easy")
    has_observation = "observation" in data
    has_reward = "reward" in data
    has_done = "done" in data
    no_info = "info" not in data
    log(f"Reset response has observation={has_observation}, reward={has_reward}, done={has_done}, no_info={no_info}",
        has_observation and has_reward and has_done and no_info)
    log(f"Reset reward is None", data["reward"] is None)
    log(f"Reset done is False", data["done"] is False)

    # Test step response format
    r = step("categorize", "T1", "technical")
    has_observation = "observation" in r
    has_reward = "reward" in r
    has_done = "done" in r
    no_info = "info" not in r
    log(f"Step response has observation={has_observation}, reward={has_reward}, done={has_done}, no_info={no_info}",
        has_observation and has_reward and has_done and no_info)
    log(f"Step reward is float", isinstance(r["reward"], (int, float)))
    log(f"Step done is bool", isinstance(r["done"], bool))

    # Test health response format
    h = requests.get(f"{BASE}/health").json()
    log(f"Health status is 'healthy'", h["status"] == "healthy")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🎫 Ticket Triage OpenEnv — Test Suite")
    print("=" * 60)

    try:
        requests.get(f"{BASE}/health", timeout=3)
    except Exception:
        print(f"\n❌ Cannot reach server at {BASE}. Is it running?")
        sys.exit(1)

    test_health_endpoints()
    test_reset()
    test_perfect_easy()
    test_perfect_medium()
    test_perfect_hard()
    test_edge_cases()
    test_state_endpoint()
    test_step_limit()
    test_double_close()
    test_openenv_response_format()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed / {FAIL} failed / {TOTAL} total")
    if FAIL == 0:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {FAIL} test(s) failed")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
