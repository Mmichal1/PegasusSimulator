#!/usr/bin/env python
"""
| File: intercept_optimizer.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Simple evolutionary optimizer for intercept controller parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from intercept_tuning import InterceptParamSpec


@dataclass
class TrialRecord:
    params: InterceptParamSpec
    score: float


class InterceptEvolutionaryOptimizer:
    """
    Lightweight self-learner:

    1. Explore random parameter vectors.
    2. Keep the best performers.
    3. Mutate elites to refine the search.

    Objective: minimize episode score (= pursuit time on hit).
    """

    def __init__(
        self,
        seed: int = 0,
        random_trials: int = 10,
        refine_trials: int = 14,
        elite_count: int = 3,
        mutation_scale: float = 0.18,
    ):
        self.rng = random.Random(seed)
        self.random_trials = int(random_trials)
        self.refine_trials = int(refine_trials)
        self.elite_count = int(elite_count)
        self.mutation_scale = float(mutation_scale)

        self._trial_index = 0
        self._records: list[TrialRecord] = []

    @property
    def total_trials(self) -> int:
        return self.random_trials + self.refine_trials

    def suggest(self) -> InterceptParamSpec:
        if self._trial_index < self.random_trials or not self._records:
            params = InterceptParamSpec.random(self.rng)
        else:
            elites = sorted(self._records, key=lambda record: record.score)[: self.elite_count]
            parent = self.rng.choice(elites).params
            params = parent.mutate(self.rng, scale=self.mutation_scale)
        self._trial_index += 1
        return params

    def tell(self, params: InterceptParamSpec, score: float) -> None:
        self._records.append(TrialRecord(params=params, score=float(score)))

    def best(self) -> tuple[InterceptParamSpec, float] | None:
        if not self._records:
            return None
        best_record = min(self._records, key=lambda record: record.score)
        return best_record.params, best_record.score

    def history(self) -> list[dict]:
        return [
            {"score": record.score, "params": record.params.to_dict()}
            for record in sorted(self._records, key=lambda record: record.score)
        ]
