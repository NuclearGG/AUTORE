"""
Inference Script — AUTORE
==========================

MANDATORY REQUIREMENTS (from organizers):
- API_BASE_URL   The API endpoint for the LLM.
- MODEL_NAME     The model identifier to use for inference.
- HF_TOKEN       Your Hugging Face / API key.

- This file is named `inference.py` and placed in the root directory.
- All LLM calls use the OpenAI Client with the above variables.
"""

import os
import re
import sys
import json
import textwrap
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from server.AUTORE_environment import AutoreEnvironment, MAX_STEPS
from tasks import run_all_tasks

# ── Environment configuration (mandatory) ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o-mini"

# ── Inference settings ─────────────────────────────────────────────────────
USE_LLM     = os.getenv("USE_LLM", "0") == "1"
TEMPERATURE = 0.0
MAX_TOKENS  = 64
FALLBACK_SIGNAL = 0   # NS green as safe fallback

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent traffic signal controller for a four-way intersection.
    You receive the current intersection state and must choose the next signal phase.

    Rules:
    - ALWAYS prioritize emergency vehicles (ambulances) — clear their lane immediately.
    - Otherwise, give green to the axis with more waiting cars.
    - Reply with ONLY a valid JSON object, no explanation, no markdown:
        {"signal": 0}   to set North-South GREEN
        {"signal": 1}   to set East-West GREEN
""").strip()

# ── Action parser ──────────────────────────────────────────────────────────
JSON_PATTERN = re.compile(r'\{[^{}]*"signal"\s*:\s*[01][^{}]*\}')


def parse_model_signal(response_text: str) -> int:
    """Extract signal (0 or 1) from model response. Falls back to FALLBACK_SIGNAL."""
    if not response_text:
        return FALLBACK_SIGNAL

    # Try structured JSON first
    match = JSON_PATTERN.search(response_text)
    if match:
        try:
            data = json.loads(match.group(0))
            return int(data.get("signal", FALLBACK_SIGNAL)) % 2
        except (json.JSONDecodeError, ValueError):
            pass

    # Try parsing full response as JSON (model might strip fences)
    clean = response_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        return int(data.get("signal", FALLBACK_SIGNAL)) % 2
    except (json.JSONDecodeError, ValueError):
        pass

    # Last resort: look for bare 0 or 1
    for token in re.findall(r"\b[01]\b", response_text):
        return int(token)

    return FALLBACK_SIGNAL


def build_user_prompt(step: int, obs: AutoreObservation) -> str:
    """Build a structured text prompt describing the current intersection state."""
    phase_label = {0: "NS (North-South) GREEN", 1: "EW (East-West) GREEN", -1: "YELLOW"}.get(
        obs.phase, "UNKNOWN"
    )
    lane_names = ["North", "South", "East", "West"]
    if obs.emergency_lane >= 0:
        emerg_str = (
            f"EMERGENCY: {lane_names[obs.emergency_lane]} lane has an ambulance — "
            "MUST be cleared this step!"
        )
    else:
        emerg_str = "No emergency vehicle."

    return textwrap.dedent(f"""
        Step: {step}
        Intersection state:
          North queue : {obs.cars_N} cars
          South queue : {obs.cars_S} cars
          East  queue : {obs.cars_E} cars
          West  queue : {obs.cars_W} cars
          Current phase : {phase_label}
          Emergency     : {emerg_str}

        Reply with ONLY a JSON object: {{"signal": 0}} or {{"signal": 1}}
    """).strip()


# ── Heuristic policy (no API required) ────────────────────────────────────

def heuristic_policy(obs: AutoreObservation) -> int:
    """
    Deterministic rule-based baseline. No API key required.

    Priority:
      1. Ambulance in N/S → NS green (0)
      2. Ambulance in E/W → EW green (1)
      3. More cars on NS  → NS green (0)
      4. Default (tie)    → NS green (0)  [arterial bias]
    """
    if obs.emergency_lane in (0, 1):
        return 0
    if obs.emergency_lane in (2, 3):
        return 1
    return 0 if (obs.cars_N + obs.cars_S) >= (obs.cars_E + obs.cars_W) else 1


# ── LLM policy ─────────────────────────────────────────────────────────────

def llm_policy_factory(client: OpenAI, model: str):
    """
    Returns a policy function that calls the LLM for every step decision.
    Falls back to heuristic_policy on any API or parse error.
    """

    def _policy(obs: AutoreObservation) -> int:
        step = getattr(obs, "_step", 0)  # informational only
        user_prompt = build_user_prompt(step, obs)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [WARN] LLM request failed ({exc}). Using heuristic fallback.")
            return heuristic_policy(obs)

        signal = parse_model_signal(response_text)
        return signal

    return _policy


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(policy_fn, seed: int = 0, verbose: bool = True) -> float:
    """Run one full episode. Returns cumulative reward."""
    env = AutoreEnvironment(seed=seed)
    result_obs = env.reset()
    total_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        signal = policy_fn(result_obs)
        result_obs = env.step(AutoreAction(signal=signal))
        total_reward += result_obs.reward

        if verbose and (step % 20 == 0 or step == 1):
            phase_str = {0: "NS", 1: "EW", -1: "YL"}.get(result_obs.phase, "??")
            emerg_flag = f" [EMERG lane={result_obs.emergency_lane}]" if result_obs.emergency_lane >= 0 else ""
            print(
                f"  step {step:3d} | "
                f"N={result_obs.cars_N:2d} S={result_obs.cars_S:2d} "
                f"E={result_obs.cars_E:2d} W={result_obs.cars_W:2d} | "
                f"phase={phase_str}{emerg_flag} | "
                f"reward={result_obs.reward:8.1f}"
            )

        if result_obs.done:
            print("  Episode complete (done flag).")
            break

    return total_reward


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 62)
    print("  AUTORE -- Autonomous Traffic Optimization & Response Engine")
    print("=" * 62)

    # Build client (always — mirrors reference script structure)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Select policy
    if USE_LLM:
        print(f"\n  Mode      : LLM")
        print(f"  Endpoint  : {API_BASE_URL}")
        print(f"  Model     : {MODEL_NAME}")
        policy = llm_policy_factory(client, MODEL_NAME)
    else:
        print(f"\n  Mode      : Heuristic  (set USE_LLM=1 to enable LLM policy)")
        policy = heuristic_policy

    # Baseline episode
    print(f"\n{'─'*62}")
    print("  Baseline Episode  (seed=0)")
    print(f"{'─'*62}")
    baseline_reward = run_episode(policy, seed=0, verbose=True)
    print(f"\n  Baseline Reward : {baseline_reward:.0f}")

    # Task graders
    print(f"\n{'─'*62}")
    print("  Task Graders")
    print(f"{'─'*62}")
    scores = run_all_tasks(policy)

    # Summary
    print(f"\n{'─'*62}")
    print("  Summary")
    print(f"{'─'*62}")
    for task, score in scores.items():
        filled = int(score * 30)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"  {task:6s}  {bar}  {score:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score   : {avg:.4f}")
    print(f"  Baseline Reward : {baseline_reward:.0f}")

    # Validate scores
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    if not all_valid:
        print("\n  ERROR: one or more scores outside [0.0, 1.0]")
        sys.exit(1)

    print("\n  All scores valid. Ready to submit.")
    print("=" * 62)
    return scores


if __name__ == "__main__":
    main()
