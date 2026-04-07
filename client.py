# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AUTORE Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AutoreAction, AutoreObservation


class AutoreEnv(EnvClient[AutoreAction, AutoreObservation, State]):
    """
    WebSocket client for the AUTORE traffic signal RL environment.

    Example:
        >>> with AutoreEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.cars_N, obs.phase)
        ...
        ...     result = env.step(AutoreAction(signal=0))   # NS green
        ...     print(result.reward)

    Example with Docker (auto-starts container):
        >>> env = AutoreEnv.from_docker_image("autore-env:latest")
        >>> try:
        ...     env.reset()
        ...     env.step(AutoreAction(signal=1))
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: AutoreAction) -> Dict:
        """Serialize AutoreAction to JSON payload for the WebSocket step message."""
        return {"signal": action.signal}

    def _parse_result(self, payload: Dict) -> StepResult[AutoreObservation]:
        """Deserialize server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = AutoreObservation(
            cars_N=obs_data.get("cars_N", 0),
            cars_S=obs_data.get("cars_S", 0),
            cars_E=obs_data.get("cars_E", 0),
            cars_W=obs_data.get("cars_W", 0),
            phase=obs_data.get("phase", 0),
            emergency_lane=obs_data.get("emergency_lane", -1),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Deserialize server response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )