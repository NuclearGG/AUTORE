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
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from server.AUTORE_environment import AutoreEnvironment, MAX_STEPS

# ── Environment variables ──────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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

def run_episode(client: OpenAI, task_name: str, seed: int = 0) -> None:
    """Run one episode, emit START/STEP/END logs."""
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if API_KEY:
                signal = get_model_signal(client, step, obs)
            else:
                signal = heuristic_policy(obs)
                
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
        # Example heuristic score calculation to keep strictly in (0, 1) bounds
        # You may adjust the normalization logic based on actual environment bounds
        base_score = max(0.0, min(1.0, (total_reward + 100000) / 100000))
        score = max(0.001, min(0.999, base_score))
        success = total_reward > -60000

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Define tasks mapped to specific environment seeds for easy, medium, hard
    tasks = {
        "easy": 0,
        "medium": 1,
        "hard": 2
    }

    for task_name, seed in tasks.items():
        run_episode(client, task_name, seed)


if __name__ == "__main__":
    main()