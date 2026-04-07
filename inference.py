"""
Inference Script — AUTORE
==========================

MANDATORY REQUIREMENTS (from organizers):
- API_BASE_URL   The API endpoint for the LLM (injected by validator).
- MODEL_NAME     The model identifier to use for inference.
- API_KEY        The LiteLLM proxy key (injected by validator).

- This file is named `inference.py` and placed in the root directory.
- All LLM calls use the OpenAI Client with base_url=os.environ["API_BASE_URL"]
  and api_key=os.environ["API_KEY"] exactly as required.
- Stdout follows the required structured format: START / STEP / END
"""

import os
import re
import sys
import json
import textwrap

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from server.AUTORE_environment import AutoreEnvironment, MAX_STEPS
from tasks import run_all_tasks

# ── Environment variables — strictly use os.environ[] as required ──────────
# Validator injects API_BASE_URL and API_KEY — read them exactly as specified.
# Fall back gracefully only for local runs where they are not injected.
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
except KeyError:
    API_BASE_URL = "https://api.openai.com/v1"

try:
    API_KEY = os.environ["API_KEY"]
except KeyError:
    API_KEY = os.environ.get("HF_TOKEN", "EMPTY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ── Auto-enable LLM when API_KEY is injected ──────────────────────────────
_has_key = bool(API_KEY) and API_KEY != "EMPTY"
USE_LLM  = _has_key or (os.environ.get("USE_LLM", "0") == "1")

# ── Inference settings ─────────────────────────────────────────────────────
TEMPERATURE     = 0.0
MAX_TOKENS      = 64
FALLBACK_SIGNAL = 0

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
    match = JSON_PATTERN.search(response_text)
    if match:
        try:
            return int(json.loads(match.group(0)).get("signal", FALLBACK_SIGNAL)) % 2
        except (json.JSONDecodeError, ValueError):
            pass
    clean = response_text.replace("```json", "").replace("```", "").strip()
    try:
        return int(json.loads(clean).get("signal", FALLBACK_SIGNAL)) % 2
    except (json.JSONDecodeError, ValueError):
        pass
    for token in re.findall(r"\b[01]\b", response_text):
        return int(token)
    return FALLBACK_SIGNAL


def build_user_prompt(step: int, obs: AutoreObservation) -> str:
    phase_label = {0: "NS (North-South) GREEN", 1: "EW (East-West) GREEN", -1: "YELLOW"}.get(
        obs.phase, "UNKNOWN"
    )
    lane_names = ["North", "South", "East", "West"]
    emerg_str = (
        f"EMERGENCY: {lane_names[obs.emergency_lane]} lane has an ambulance — MUST be cleared!"
        if obs.emergency_lane >= 0
        else "No emergency vehicle."
    )
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


# ── Heuristic policy ───────────────────────────────────────────────────────

def heuristic_policy(obs: AutoreObservation) -> int:
    """Deterministic rule-based baseline. Used as fallback when LLM fails."""
    if obs.emergency_lane in (0, 1):
        return 0
    if obs.emergency_lane in (2, 3):
        return 1
    return 0 if (obs.cars_N + obs.cars_S) >= (obs.cars_E + obs.cars_W) else 1


# ── LLM policy ─────────────────────────────────────────────────────────────

def llm_policy_factory(client: OpenAI, model: str):
    """Returns a policy that calls the LLM each step. Falls back to heuristic on error."""

    def _policy(obs: AutoreObservation) -> int:
        step = getattr(obs, "_step", 0)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": build_user_prompt(step, obs)}]},
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
            print(f"[WARN] LLM request failed ({exc}). Using heuristic fallback.")
            return heuristic_policy(obs)
        return parse_model_signal(response_text)

    return _policy


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(policy_fn, seed: int = 0, episode: int = 0) -> float:
    """Run one full episode. Emits START, STEP, and END log lines."""
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()
    total_reward = 0.0

    print(f"[START] episode={episode} seed={seed} max_steps={MAX_STEPS}")

    for step in range(1, MAX_STEPS + 1):
        signal = policy_fn(obs)
        obs = env.step(AutoreAction(signal=signal))
        total_reward += obs.reward

        print(
            f"[STEP] episode={episode} step={step} "
            f"signal={signal} "
            f"phase={obs.phase} "
            f"N={obs.cars_N} S={obs.cars_S} E={obs.cars_E} W={obs.cars_W} "
            f"emergency={obs.emergency_lane} "
            f"reward={obs.reward:.2f} "
            f"total_reward={total_reward:.2f} "
            f"done={obs.done}"
        )

        if obs.done:
            break

    print(f"[END] episode={episode} total_reward={total_reward:.2f} steps={env.state.step_count}")
    return total_reward


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> dict:
    print("[INFO] AUTORE -- Autonomous Traffic Optimization & Response Engine")
    print(f"[INFO] API_BASE_URL={API_BASE_URL}")
    print(f"[INFO] MODEL_NAME={MODEL_NAME}")
    print(f"[INFO] USE_LLM={USE_LLM}")
    print(f"[INFO] API_KEY={'SET' if _has_key else 'NOT SET (heuristic mode)'}")

    # Initialise OpenAI client exactly as the validator requires:
    # base_url=os.environ["API_BASE_URL"]  and  api_key=os.environ["API_KEY"]
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Select policy
    if USE_LLM:
        print("[INFO] Mode: LLM")
        policy = llm_policy_factory(client, MODEL_NAME)
    else:
        print("[INFO] Mode: Heuristic (API_KEY not set)")
        policy = heuristic_policy

    # Baseline episode
    print("[INFO] Running baseline episode...")
    baseline_reward = run_episode(policy, seed=0, episode=0)

    # Task graders
    print("[INFO] Running task graders...")
    scores = run_all_tasks(policy)

    # Summary
    print("[SCORES] " + json.dumps(scores))
    print(f"[SCORES] average={sum(scores.values()) / len(scores):.4f}")
    print(f"[SCORES] baseline_reward={baseline_reward:.2f}")

    # Validate
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    if not all_valid:
        print("[ERROR] One or more scores outside [0.0, 1.0]")
        sys.exit(1)

    print("[INFO] All scores valid. Ready to submit.")
    return scores


if __name__ == "__main__":
    main()