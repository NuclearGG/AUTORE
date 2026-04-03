# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AUTORE Environment.

Observation space (6-dim):
    [cars_N, cars_S, cars_E, cars_W, phase, emergency_lane]

Action space (Discrete 2):
    0 → North-South green
    1 → East-West green
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AutoreAction(Action):
    """
    Traffic signal action.

    Attributes:
        signal: 0 = set NS green, 1 = set EW green
    """

    signal: int = Field(
        ...,
        ge=0,
        le=1,
        description="Signal phase: 0 = North-South green, 1 = East-West green",
    )


class AutoreObservation(Observation):
    """
    Intersection state observation.

    Attributes:
        cars_N: Vehicles queued in North lane
        cars_S: Vehicles queued in South lane
        cars_E: Vehicles queued in East lane
        cars_W: Vehicles queued in West lane
        phase:  Current signal phase (0=NS green, 1=EW green, -1=yellow)
        emergency_lane: Lane index of ambulance (-1 if none, 0=N 1=S 2=E 3=W)
    """

    cars_N: int = Field(default=0, ge=0, description="Vehicles queued in North lane")
    cars_S: int = Field(default=0, ge=0, description="Vehicles queued in South lane")
    cars_E: int = Field(default=0, ge=0, description="Vehicles queued in East lane")
    cars_W: int = Field(default=0, ge=0, description="Vehicles queued in West lane")
    phase: int = Field(
        default=0,
        description="Signal phase: 0=NS green, 1=EW green, -1=yellow",
    )
    emergency_lane: int = Field(
        default=-1,
        description="Lane index of ambulance (0=N,1=S,2=E,3=W) or -1 if none",
    )

    @property
    def observation_vector(self) -> list:
        """Return the 6-dim observation as a plain list."""
        return [
            self.cars_N,
            self.cars_S,
            self.cars_E,
            self.cars_W,
            self.phase,
            self.emergency_lane,
        ]