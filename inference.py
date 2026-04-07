"""
Inference Script — AUTORE
===================================
This script connects to the AUTORE environment, runs a traffic signal control task,
and logs structured outputs for evaluation.

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
"""

import asyncio
import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from client import AutoreEnv

# ── Environment variables ──────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME  = os.getenv("AUTORE_TASK", "traffic-signal-control")
BENCHMARK  = os.getenv("AUTORE_BENCHMARK", "AUTORE")
MAX_STEPS  = 120
TEMPERATURE = 0.7
MAX_TOKENS  = 64
SUCCESS_SCORE_THRESHOLD = 0.3   # normalized score in [0, 1]

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


# ── Prompt builder ─────────────────────────────────────────────────────────

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


# ── LLM call ───────────────────────────────────────────────────────────────

def get_model_signal(client: OpenAI, step: int, obs: AutoreObservation) -> int:
    user_prompt = build_user_prompt(step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse JSON response
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return int(data.get("signal", 0)) % 2
    except Exception as exc:
        print(f"[CRITICAL ERROR] Model request failed: {exc}", flush=True)
        # Raise the error to fail loudly so the platform logs capture it
        raise exc


# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment via Docker image or direct URL
    if IMAGE_NAME:
        env = await AutoreEnv.from_docker_image(IMAGE_NAME)
    else:
        hf_space_url = os.getenv(
            "ENV_URL", "https://nucleargg-autore.hf.space"
        )
        env = AutoreEnv(base_url=hf_space_url)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        # Initialize tracking variables OUTSIDE the loop
        current_active_signal = 0
        time_in_phase = 0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # --- ANTI-STARVATION OVERRIDE ---
            # If the light is green for 8+ steps and there is no ambulance, FORCE a switch
            if time_in_phase >= 8 and obs.emergency_lane == -1:
                signal = 1 if current_active_signal == 0 else 0
                print(f"[DEBUG] Step {step}: Forced switch to prevent lane starvation.", flush=True)
            else:
                # Otherwise, let the LLM decide
                signal = get_model_signal(client, step, obs)

            # --- UPDATE TRACKERS WITH NEW SIGNAL ---
            if signal == current_active_signal:
                time_in_phase += 1
            else:
                current_active_signal = signal
                time_in_phase = 0

            action_str = f"signal={signal}"

            result = await env.step(AutoreAction(signal=signal))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Normalize score to [0, 1]
        total_reward = sum(rewards)
        WORST = -72000.0
        BEST  = -46000.0
        score = max(0.0, min(1.0, (total_reward - WORST) / (BEST - WORST)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[CRITICAL ERROR] Episode error: {e}", flush=True)
        raise e # Let the exception crash the run so validator sees it

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())