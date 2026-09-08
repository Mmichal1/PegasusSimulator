#!/usr/bin/env python
"""
| File: intercept_episode_runner.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Run finite intercept episodes and measure time-to-hit.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import carb
except ImportError:  # pragma: no cover
    carb = None

from intercept_tuning import (
    DEFAULT_HIT_RADIUS_M,
    DEFAULT_MAX_EPISODE_TIME_S,
    DEFAULT_MIN_PROGRESS_M,
    DEFAULT_STALL_TIMEOUT_S,
    DEFAULT_STARTUP_DELAY_S,
    EpisodeResult,
    InterceptParamSpec,
    apply_intercept_params,
)


class InterceptEpisodeRunner:
    def __init__(
        self,
        world: Any,
        timeline: Any,
        pursuer_controller: Any,
        target_controller: Any,
        tracker: Any,
        pursuer_path: str = "/World/pursuer_drone",
        target_path: str = "/World/target_drone",
        hit_radius_m: float = DEFAULT_HIT_RADIUS_M,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        max_episode_time_s: float = DEFAULT_MAX_EPISODE_TIME_S,
        min_progress_m: float = DEFAULT_MIN_PROGRESS_M,
        startup_delay_s: float = DEFAULT_STARTUP_DELAY_S,
        physics_dt: float = 1.0 / 500.0,
    ):
        self.world = world
        self.timeline = timeline
        self.pursuer_controller = pursuer_controller
        self.target_controller = target_controller
        self.tracker = tracker
        self.pursuer_path = pursuer_path
        self.target_path = target_path
        self.hit_radius_m = float(hit_radius_m)
        self.stall_timeout_s = float(stall_timeout_s)
        self.max_episode_time_s = float(max_episode_time_s)
        self.min_progress_m = float(min_progress_m)
        self.startup_delay_s = float(startup_delay_s)
        self.physics_dt = float(physics_dt)

    def _format_progress_message(
        self,
        wall_elapsed_s: float,
        sim_time_s: float,
        stall_time_left_s: float,
        best_distance_m: float,
        extra: str = "",
    ) -> str:
        best_label = f"{best_distance_m:.2f}m" if np.isfinite(best_distance_m) else "n/a"
        suffix = f" {extra}" if extra else ""
        return (
            f"[wall {wall_elapsed_s:6.1f}s | sim {sim_time_s:5.1f}s] "
            f"stall timeout {max(0.0, stall_time_left_s):5.1f}s "
            f"(best {best_label}){suffix}"
        )

    def _log_episode_progress(self, message: str) -> None:
        if carb is not None:
            carb.log_info(message)
        print(message, flush=True)

    def _get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        from pegasus.simulator.logic.vehicle_manager import VehicleManager

        manager = VehicleManager.get_vehicle_manager()
        pursuer = manager.get_vehicle(self.pursuer_path)
        target = manager.get_vehicle(self.target_path)
        if pursuer is None or target is None:
            raise RuntimeError("Pursuer or target vehicle not found in VehicleManager")
        return (
            np.asarray(pursuer.state.position, dtype=float),
            np.asarray(target.state.position, dtype=float),
        )

    def _register_progress(self, distance: float, best_distance: float, stall_time_left: float) -> tuple[float, float]:
        if not np.isfinite(best_distance):
            return distance, self.stall_timeout_s

        if distance + self.min_progress_m <= best_distance:
            return distance, self.stall_timeout_s

        return best_distance, stall_time_left

    def reset_episode(self) -> None:
        self.timeline.stop()
        self.world.reset()

        if hasattr(self.pursuer_controller, "reset_episode"):
            self.pursuer_controller.reset_episode()
        if hasattr(self.target_controller, "reset_episode"):
            self.target_controller.reset_episode()
        elif hasattr(self.target_controller, "reset_statistics"):
            self.target_controller.reset_statistics()

        if hasattr(self.tracker, "_tracker"):
            self.tracker._tracker.locked_target_id = None

    def run_episode(self, params: InterceptParamSpec) -> EpisodeResult:
        apply_intercept_params(self.pursuer_controller, params)
        self.reset_episode()
        self.timeline.play()

        wall_start = time.perf_counter()
        sim_time = 0.0
        min_distance = float("inf")
        best_distance = float("inf")
        stall_time_left = self.stall_timeout_s
        hit = False
        time_to_hit = 0.0
        end_reason = "stall"
        next_sim_log_second = 1

        self._log_episode_progress(
            self._format_progress_message(
                0.0,
                0.0,
                stall_time_left,
                best_distance,
                extra="episode started",
            )
        )

        while sim_time < self.max_episode_time_s:
            self.world.step(render=True)
            sim_time += self.physics_dt
            stall_time_left -= self.physics_dt

            pursuer_p, target_p = self._get_positions()
            distance = float(np.linalg.norm(pursuer_p - target_p))
            min_distance = min(min_distance, distance)

            previous_best = best_distance
            best_distance, stall_time_left = self._register_progress(
                distance, best_distance, stall_time_left
            )
            if np.isfinite(previous_best) and best_distance < previous_best:
                wall_elapsed = time.perf_counter() - wall_start
                self._log_episode_progress(
                    self._format_progress_message(
                        wall_elapsed,
                        sim_time,
                        stall_time_left,
                        best_distance,
                        extra=f"new best, reset to {self.stall_timeout_s:.1f}s",
                    )
                )

            while sim_time >= next_sim_log_second:
                wall_elapsed = time.perf_counter() - wall_start
                self._log_episode_progress(
                    self._format_progress_message(
                        wall_elapsed,
                        float(next_sim_log_second),
                        stall_time_left,
                        best_distance,
                    )
                )
                next_sim_log_second += 1

            if distance <= self.hit_radius_m:
                hit = True
                time_to_hit = sim_time
                end_reason = "hit"
                break

            if stall_time_left <= 0.0:
                end_reason = "stall"
                break

        if sim_time >= self.max_episode_time_s and not hit:
            end_reason = "safety_cap"

        pursuer_p, target_p = self._get_positions()
        final_distance = float(np.linalg.norm(pursuer_p - target_p))
        pursuit_time = max(0.0, time_to_hit - self.startup_delay_s) if hit else sim_time

        self.timeline.stop()

        return EpisodeResult(
            hit=hit,
            time_to_hit_s=time_to_hit if hit else sim_time,
            pursuit_time_s=pursuit_time if hit else max(0.0, sim_time - self.startup_delay_s),
            min_distance_m=min_distance if np.isfinite(min_distance) else final_distance,
            final_distance_m=final_distance,
            episode_time_s=sim_time,
            end_reason=end_reason,
            params=params.to_dict(),
        )
