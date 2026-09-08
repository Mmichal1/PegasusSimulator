#!/usr/bin/env python
"""
| File: intercept_tuning.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Parameter space, scoring, and application for intercept self-tuning.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HIT_RADIUS_M = 1.0
DEFAULT_STALL_TIMEOUT_S = 50.0
DEFAULT_MAX_EPISODE_TIME_S = 300.0
DEFAULT_MIN_PROGRESS_M = 0.05
DEFAULT_STARTUP_DELAY_S = 5.0


@dataclass
class InterceptParamSpec:
    intercept_lateral_gain: float = 0.35
    intercept_climb_gain: float = 0.25
    velocity_feedforward_gain: float = 0.45
    climb_speed: float = 2.0
    speed_boost_factor: float = 1.2
    max_horizontal_speed: float = 1.0
    command_velocity_smoothing: float = 0.1
    centroid_smoothing: float = 0.15
    velocity_smoothing: float = 0.12
    prediction_smoothing: float = 0.15
    max_prediction_offset_px: float = 70.0
    kp_z: float = 10.0
    kd_z: float = 11.0

    @classmethod
    def bounds(cls) -> dict[str, tuple[float, float]]:
        return {
            "intercept_lateral_gain": (0.15, 1.0),
            "intercept_climb_gain": (0.05, 0.8),
            "velocity_feedforward_gain": (0.0, 1.2),
            "climb_speed": (1.2, 4.0),
            "speed_boost_factor": (1.0, 2.0),
            "max_horizontal_speed": (0.4, 2.5),
            "command_velocity_smoothing": (0.04, 0.25),
            "centroid_smoothing": (0.06, 0.3),
            "velocity_smoothing": (0.05, 0.25),
            "prediction_smoothing": (0.05, 0.35),
            "max_prediction_offset_px": (25.0, 110.0),
            "kp_z": (5.0, 14.0),
            "kd_z": (5.0, 14.0),
        }

    @classmethod
    def random(cls, rng: random.Random | None = None) -> InterceptParamSpec:
        rng = rng or random.Random()
        bounds = cls.bounds()
        values = {name: rng.uniform(low, high) for name, (low, high) in bounds.items()}
        return cls(**values)

    def mutate(self, rng: random.Random, scale: float = 0.2) -> InterceptParamSpec:
        bounds = self.bounds()
        mutated: dict[str, float] = {}
        for name, value in asdict(self).items():
            low, high = bounds[name]
            span = high - low
            delta = rng.uniform(-scale, scale) * span
            mutated[name] = float(np.clip(value + delta, low, high))
        return InterceptParamSpec(**mutated)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class EpisodeResult:
    hit: bool
    time_to_hit_s: float
    pursuit_time_s: float
    min_distance_m: float
    final_distance_m: float
    episode_time_s: float
    end_reason: str
    params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "time_to_hit_s": self.time_to_hit_s,
            "pursuit_time_s": self.pursuit_time_s,
            "min_distance_m": self.min_distance_m,
            "final_distance_m": self.final_distance_m,
            "episode_time_s": self.episode_time_s,
            "end_reason": self.end_reason,
            "params": self.params,
            "score": score_episode(self),
        }


def score_episode(
    result: EpisodeResult,
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
) -> float:
    """
    Lower is better.

    Successful intercepts are scored by pursuit time (after startup hold).
    Misses are penalized with episode duration plus closest-approach distance.
    """
    if result.hit:
        return float(result.pursuit_time_s)
    return float(result.episode_time_s + result.min_distance_m * 5.0 + stall_timeout_s)


def apply_intercept_params(controller: Any, params: InterceptParamSpec) -> None:
    controller.k_lateral = float(params.intercept_lateral_gain)
    controller.intercept_climb_gain = float(params.intercept_climb_gain)
    controller.velocity_feedforward_gain = float(params.velocity_feedforward_gain)
    controller.climb_speed = float(params.climb_speed)
    controller.speed_boost_factor = float(params.speed_boost_factor)
    controller.max_horizontal_speed = float(params.max_horizontal_speed)
    controller.command_velocity_smoothing = float(params.command_velocity_smoothing)
    controller.centroid_smoothing = float(params.centroid_smoothing)

    predictor = controller._predictor
    predictor.velocity_smoothing = float(params.velocity_smoothing)
    predictor.prediction_smoothing = float(params.prediction_smoothing)
    predictor.max_prediction_offset_px = float(params.max_prediction_offset_px)
    predictor.pursuer_speed_mps = float(params.climb_speed * params.speed_boost_factor)

    controller.Kp[2, 2] = float(params.kp_z)
    controller.Kd[2, 2] = float(params.kd_z)


def save_best_params(
    params: InterceptParamSpec,
    score: float,
    result: EpisodeResult,
    output_path: str | Path,
) -> None:
    payload = {
        "score": float(score),
        "params": params.to_dict(),
        "result": result.to_dict(),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_best_params(path: str | Path) -> InterceptParamSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InterceptParamSpec(**payload["params"])
