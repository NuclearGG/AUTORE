"""
AUTORE Task Graders
===================

Three tasks evaluate agent performance on progressively harder aspects
of the AUTORE traffic signal control problem.

All scores are normalized to [0.0, 1.0] — deterministic, reproducible.

Tasks
-----
  easy    Minimize total waiting cars across the full episode (120 steps)
  medium  Keep congestion low during the rush-hour window   (steps 40-80)
  hard    Respond immediately to emergency vehicles

Scoring
-------
Easy / Medium use reward-anchored normalization:
    score = clamp((agent_reward - WORST) / (BEST - WORST), 0, 1)

  Anchors were derived empirically across 20 fixed seeds:
    WORST = always-EW policy on NS-heavy roads  (~-70 000 cumulative reward)
    BEST  = reference heuristic                 (~-46 000 cumulative reward)

Hard uses a direct ratio:
    score = steps_where_ambulance_was_cleared / steps_where_ambulance_was_present

Usage
-----
    from tasks import run_all_tasks
    scores = run_all_tasks(policy_fn)   # policy_fn(obs) -> int

Or standalone:
    python tasks.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AutoreAction, AutoreObservation
from server.AUTORE_environment import (
    AutoreEnvironment,
    MAX_STEPS,
    RUSH_HOUR_START_STEP,
    RUSH_HOUR_END_STEP,
    PHASE_NS_GREEN,
    PHASE_EW_GREEN,
    PHASE_YELLOW,
)

# ── Calibration anchors ────────────────────────────────────────────────────
# Easy  — full-episode cumulative reward
EASY_WORST:   float = -72_000.0   # always-EW policy (worst for heavy NS roads)
EASY_BEST:    float = -46_000.0   # reference heuristic

# Medium — reward accumulated only during rush-hour window (steps 40-80)
MEDIUM_WORST: float = -24_000.0
MEDIUM_BEST:  float = -15_000.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.001, min(0.999, float(v)))


def _normalize(value: float, worst: float, best: float) -> float:
    span = best - worst
    if abs(span) < 1e-9:
        return 0.5
    return _clamp((value - worst) / span)


# ── Task 1 — Easy ──────────────────────────────────────────────────────────

def grade_easy(policy_fn, seed: int = 0) -> float:
    """
    Objective: Minimize total waiting cars across the full 120-step episode.

    Returns normalized score in [0, 1]. Higher is better.
    """
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        obs = env.step(AutoreAction(signal=policy_fn(obs)))
        total_reward += obs.reward

    return _normalize(total_reward, EASY_WORST, EASY_BEST)


# ── Task 2 — Medium ────────────────────────────────────────────────────────

def grade_medium(policy_fn, seed: int = 1) -> float:
    """
    Objective: Keep congestion low during the rush-hour window (steps 40-80).

    Returns normalized score in [0, 1]. Higher is better.
    """
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()
    rush_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        obs = env.step(AutoreAction(signal=policy_fn(obs)))
        if RUSH_HOUR_START_STEP <= step <= RUSH_HOUR_END_STEP:
            rush_reward += obs.reward

    return _normalize(rush_reward, MEDIUM_WORST, MEDIUM_BEST)


# ── Task 3 — Hard ──────────────────────────────────────────────────────────

def grade_hard(policy_fn, seed: int = 2) -> float:
    """
    Objective: Respond to every ambulance within one step.

    score = cleared_steps / present_steps
      cleared_steps = steps where ambulance was present at start AND gone by end
      present_steps = steps where ambulance was present at start

    Returns float in [0, 1].
      1.0 = every ambulance cleared on the same step it appeared
      0.5 = no ambulance appeared (neutral)
      0.0 = every ambulance was blocked all episode
    """
    env = AutoreEnvironment(seed=seed)
    obs = env.reset()

    present_steps = 0
    cleared_steps = 0

    for _ in range(MAX_STEPS):
        had_emergency = obs.emergency_lane != -1
        obs = env.step(AutoreAction(signal=policy_fn(obs)))

        if had_emergency:
            present_steps += 1
            if obs.emergency_lane == -1:   # cleared this step
                cleared_steps += 1

    if present_steps == 0:
        return 0.5   # neutral — no ambulance spawned this seed

    return _clamp(cleared_steps / present_steps)


# ── Registry ───────────────────────────────────────────────────────────────

TASKS: dict = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def run_all_tasks(policy_fn) -> dict:
    """
    Run all three graders and return a score dict.

    Args:
        policy_fn: callable(AutoreObservation) -> int  (0 or 1)

    Returns:
        {"easy": float, "medium": float, "hard": float}
        All values in [0.0, 1.0].
    """
    scores = {}
    for name, grader in TASKS.items():
        score = grader(policy_fn)
        assert 0.0 < score < 1.0, f"Grader '{name}' produced out-of-range score: {score}"
        scores[name] = score
        print(f"  [{name:6s}]  score = {score:.4f}")
    return scores


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from inference import heuristic_policy
    except ImportError:
        def heuristic_policy(obs):
            if obs.emergency_lane in (0, 1): return 0
            if obs.emergency_lane in (2, 3): return 1
            return 0 if (obs.cars_N + obs.cars_S) >= (obs.cars_E + obs.cars_W) else 1

    print("AUTORE task graders — heuristic policy\n")
    scores = run_all_tasks(heuristic_policy)
    avg = sum(scores.values()) / len(scores)
    print(f"\nAverage : {avg:.4f}")
    all_valid = all(0.0 < s < 1.0 for s in scores.values())
    print("Scores in [0,1]:", "OK" if all_valid else "FAIL")
    if not all_valid:
        sys.exit(1)