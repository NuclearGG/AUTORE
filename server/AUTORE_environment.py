# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AUTORE — Autonomous Traffic Optimization & Response Engine.

A realistic reinforcement learning environment simulating intelligent
traffic signal control at a four-way intersection with:

  - Asymmetric traffic flow  (heavy NS arterial vs lighter EW side roads)
  - Rush-hour congestion     (×2.5 traffic spike mid-episode)
  - Emergency vehicle handling (ambulance must be cleared immediately)

Observation (6-dim vector):
    [cars_N, cars_S, cars_E, cars_W, phase, emergency_lane]

Action (Discrete 2):
    0 → set North-South green
    1 → set East-West green

Reward per step:
    - total_waiting_cars                   (always negative)
    ×3 multiplier during yellow transition
    −50 per step while an ambulance is blocked
"""

import random
import math
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AutoreAction, AutoreObservation
except ImportError:
    from models import AutoreAction, AutoreObservation

# ── Constants ──────────────────────────────────────────────────────────────

# Signal phases
PHASE_NS_GREEN: int =  0   # North-South green, East-West red
PHASE_EW_GREEN: int =  1   # East-West green, North-South red
PHASE_YELLOW:   int = -1   # Yellow / transition (1 step)

# Poisson arrival rates (cars per step, λ)
BASE_RATE_NS: float = 3.0  # Arterial road — heavy
BASE_RATE_EW: float = 1.2  # Side streets  — light

# Rush-hour parameters
RUSH_HOUR_MULTIPLIER: float = 2.5
RUSH_HOUR_START_STEP: int   = 40
RUSH_HOUR_END_STEP:   int   = 80

# Emergency vehicle
EMERGENCY_PROB: float = 0.04   # 4 % chance per step of a new ambulance

# Cars discharged per step on the green axis (per lane)
DISCHARGE_PER_LANE: int = 3

# Reward penalties
YELLOW_PENALTY_MULT:    int   = 3
EMERGENCY_BLOCK_PENALTY: float = 50.0

# Episode length
MAX_STEPS: int = 8

# Lane indices
LANE_N, LANE_S, LANE_E, LANE_W = 0, 1, 2, 3


# ── Environment ────────────────────────────────────────────────────────────

class AutoreEnvironment(Environment):
    """
    Intelligent traffic signal RL environment.

    Each call to reset() starts a new episode with a fresh random seed
    drawn from an internal counter, ensuring reproducible but varied episodes.
    Pass `seed` to __init__ to pin the sequence for deterministic evaluation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 0):
        self._base_seed = seed
        self._episode_count = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(seed)
        self._init_traffic_state()

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self) -> AutoreObservation:
        """Reset environment to initial state and return the first observation."""
        self._episode_count += 1
        # Each episode gets its own reproducible RNG derived from base seed
        self._rng = random.Random(self._base_seed + self._episode_count)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_traffic_state()
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: AutoreAction) -> AutoreObservation:
        """
        Advance one timestep.

        Args:
            action: AutoreAction — signal field: 0=NS green, 1=EW green

        Returns:
            AutoreObservation with updated state and reward
        """
        self._state.step_count += 1
        step = self._state.step_count
        desired = action.signal  # 0 or 1

        # ── Phase transition (yellow insertion) ────────────────────
        if self._yellow_remaining > 0:
            # Locked in yellow — ignore action request
            self._phase = PHASE_YELLOW
            self._yellow_remaining -= 1
            # Yellow just expired → commit to queued green on THIS step
            if self._yellow_remaining == 0:
                self._committed_green = self._next_green
                self._phase = self._committed_green
        elif desired != self._committed_green:
            # Phase change requested → insert one yellow step
            self._phase = PHASE_YELLOW
            self._yellow_remaining = 1
            self._next_green = desired
        else:
            # Stay on current green
            self._phase = self._committed_green

        # ── Traffic arrivals (Poisson) ─────────────────────────────
        mult = self._rush_multiplier(step)
        self._queues[LANE_N] += self._poisson(BASE_RATE_NS * mult)
        self._queues[LANE_S] += self._poisson(BASE_RATE_NS * mult)
        self._queues[LANE_E] += self._poisson(BASE_RATE_EW * mult)
        self._queues[LANE_W] += self._poisson(BASE_RATE_EW * mult)

        # ── Discharge (green lanes only) ───────────────────────────
        if self._phase == PHASE_NS_GREEN:
            self._queues[LANE_N] = max(0, self._queues[LANE_N] - DISCHARGE_PER_LANE)
            self._queues[LANE_S] = max(0, self._queues[LANE_S] - DISCHARGE_PER_LANE)
        elif self._phase == PHASE_EW_GREEN:
            self._queues[LANE_E] = max(0, self._queues[LANE_E] - DISCHARGE_PER_LANE)
            self._queues[LANE_W] = max(0, self._queues[LANE_W] - DISCHARGE_PER_LANE)
        # Yellow → no discharge

        # ── Emergency vehicle ──────────────────────────────────────
        # Spawn new ambulance if none active
        if self._emergency_lane == -1 and self._rng.random() < EMERGENCY_PROB:
            self._emergency_lane = self._rng.randint(0, 3)

        # Clear ambulance if its axis is now green
        if self._emergency_lane in (LANE_N, LANE_S) and self._phase == PHASE_NS_GREEN:
            self._emergency_lane = -1
        elif self._emergency_lane in (LANE_E, LANE_W) and self._phase == PHASE_EW_GREEN:
            self._emergency_lane = -1

        # ── Reward ────────────────────────────────────────────────
        total_waiting = sum(self._queues)
        reward = -float(total_waiting)

        if self._phase == PHASE_YELLOW:
            reward *= YELLOW_PENALTY_MULT

        if self._emergency_lane != -1:
            reward -= EMERGENCY_BLOCK_PENALTY

        done = (step >= MAX_STEPS)
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ───────────────────────────────────────────────────

    def _init_traffic_state(self):
        self._queues: list = [0, 0, 0, 0]
        self._phase: int = PHASE_NS_GREEN
        self._committed_green: int = PHASE_NS_GREEN
        self._yellow_remaining: int = 0
        self._next_green: int = PHASE_NS_GREEN
        self._emergency_lane: int = -1

    def _rush_multiplier(self, step: int) -> float:
        return RUSH_HOUR_MULTIPLIER if RUSH_HOUR_START_STEP <= step <= RUSH_HOUR_END_STEP else 1.0

    def _poisson(self, lam: float) -> int:
        """Sample from Poisson(λ) using Knuth's algorithm."""
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    def _build_observation(self, reward: float, done: bool) -> AutoreObservation:
        return AutoreObservation(
            cars_N=self._queues[LANE_N],
            cars_S=self._queues[LANE_S],
            cars_E=self._queues[LANE_E],
            cars_W=self._queues[LANE_W],
            phase=self._phase,
            emergency_lane=self._emergency_lane,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "rush_hour": self._rush_multiplier(self._state.step_count) > 1.0,
                "total_waiting": sum(self._queues),
            },
        )


# ── Quick local smoke-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    env = AutoreEnvironment(seed=42)
    obs = env.reset()
    print(f"Reset  → phase={obs.phase}  queues=[{obs.cars_N},{obs.cars_S},{obs.cars_E},{obs.cars_W}]")

    total_reward = 0.0
    for t in range(MAX_STEPS):
        if obs.emergency_lane in (0, 1):
            sig = 0
        elif obs.emergency_lane in (2, 3):
            sig = 1
        elif obs.cars_N + obs.cars_S >= obs.cars_E + obs.cars_W:
            sig = 0
        else:
            sig = 1

        obs = env.step(AutoreAction(signal=sig))
        total_reward += obs.reward

    print(f"Done   → steps={env.state.step_count}  total_reward={total_reward:.0f}")
    print(f"Final queues: N={obs.cars_N} S={obs.cars_S} E={obs.cars_E} W={obs.cars_W}")