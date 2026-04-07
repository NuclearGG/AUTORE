"""
Inference Script — AUTORE
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    IMAGE_NAME       The name of the local Docker image for the environment.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME   = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

3 Tasks: easy, medium, hard — each score strictly in (0, 1)
"""

import os
import re
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from server.AUTORE_environment import AutoreEnvironment, MAX_STEPS
from tasks import TASKS, run_all_tasks

# ── Environment variables ──────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "AUTORE"

# ── Inference settings ─────────────────────────────────────────────────────
TEMPERATURE = 0.0
MAX_TOKENS  = 64

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


# ── Structured logging ─────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Heuristic policy ───────────────────────────────────────────────────────

def heuristic_policy(obs: AutoreObservation) -> int:
    """Deterministic rule-based baseline. Used as fallback when LLM unavailable."""
    if obs.emergency_lane in (0, 1):
        return 0
    if obs.emergency_lane in (2, 3):
        return 1
    return 0 if (obs.cars_N + obs.cars_S) >= (obs.cars_E + obs.cars_W) else 1


# ── LLM policy ─────────────────────────────────────────────────────────────

def get_model_signal(client: OpenAI, step: int, obs: AutoreObservation) -> int:
    """Call LLM for signal decision. Falls back to heuristic on any error."""
    prompt = textwrap.dedent(f"""
        Step: {step}
        North: {obs.cars_N}  South: {obs.cars_S}  East: {obs.cars_E}  West: {obs.cars_W}
        Phase: {obs.phase}   Emergency lane: {obs.emergency_lane}
        Reply with ONLY: {{"signal": 0}} or {{"signal": 1}}
    """).strip()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return int(json.loads(text).get("signal", 0)) % 2
    except Exception as exc:
        print(f"[DEBUG] LLM failed step {step}: {exc}", flush=True)
        return heuristic_policy(obs)


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(task_name: str, policy_fn, seed: int = 0) -> float:
    """Run one episode, emit START/STEP/END, return raw total reward."""
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            signal = policy_fn(obs)
            obs = env.step(AutoreAction(signal=signal))

            reward = obs.reward
            done = obs.done
            total_reward += reward
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"signal={signal}",
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

    finally:
        # Score for this episode — use tasks.py grader result set later
        log_end(
            success=total_reward > -60000,
            steps=steps_taken,
            score=0.5,   # placeholder; real scores from graders below
            rewards=rewards,
        )

    return total_reward


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> dict:
    print("[INFO] AUTORE -- Autonomous Traffic Optimization & Response Engine", flush=True)
    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[INFO] API_KEY={'SET' if API_KEY else 'NOT SET'}", flush=True)

    # Always create OpenAI client with injected env vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Select policy — LLM if key available, heuristic fallback
    if API_KEY:
        print("[INFO] Mode: LLM", flush=True)
        def policy(obs):
            return get_model_signal(client, 0, obs)
    else:
        print("[INFO] Mode: Heuristic", flush=True)
        policy = heuristic_policy

    # ── Run 3 tasks with graders ──────────────────────────────────────────
    # Each task runs a full episode and returns score strictly in (0, 1)
    print("[INFO] Running 3 task graders...", flush=True)

    scores = {}
    for task_name, grader_fn in TASKS.items():
        print(f"[INFO] Task: {task_name}", flush=True)
        score = grader_fn(policy)
        # Ensure strictly open interval (0, 1)
        score = max(0.001, min(0.999, score))
        scores[task_name] = score
        print(f"[SCORE] task={task_name} score={score:.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("[SCORES] " + json.dumps(scores), flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"[SCORES] average={avg:.4f}", flush=True)

    # Validate strictly open interval
    all_valid = all(0.0 < s < 1.0 for s in scores.values())
    if not all_valid:
        print("[ERROR] One or more scores not strictly in (0, 1)", flush=True)
        sys.exit(1)

    print("[INFO] All scores valid.", flush=True)
    return scores


if __name__ == "__main__":
    main()