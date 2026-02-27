# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small scheduling library for scalar coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type

__all__ = ["Schedule", "LinearSchedule", "build_schedule", "list_schedules"]


class Schedule:
    """Base class for scalar schedules."""

    def value_at(self, step: int) -> float:
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.value_at(step)


@dataclass
class LinearSchedule(Schedule):
    start: float
    end: float
    total_steps: int

    def value_at(self, step: int) -> float:
        if self.total_steps <= 1:
            return float(self.end)
        step = max(1, min(int(step), int(self.total_steps)))
        progress = (step - 1) / (self.total_steps - 1)
        return float(self.start + (self.end - self.start) * progress)


_SCHEDULE_REGISTRY: Dict[str, Type[Schedule]] = {
    "linear": LinearSchedule,
}


def build_schedule(name: str, start: float, end: float, total_steps: int) -> Schedule:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not name:
        raise ValueError("scheduling_algo must be set when schedule is enabled.")
    key = str(name).lower()
    if key not in _SCHEDULE_REGISTRY:
        available = ", ".join(sorted(_SCHEDULE_REGISTRY))
        raise ValueError(f"Unknown scheduling_algo '{name}'. Available: {available}")
    return _SCHEDULE_REGISTRY[key](start=float(start), end=float(end), total_steps=int(total_steps))


def list_schedules() -> List[str]:
    return sorted(_SCHEDULE_REGISTRY.keys())
