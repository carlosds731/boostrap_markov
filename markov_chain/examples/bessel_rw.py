import math
import os
import numpy as np
from markov_chain import MarkovChain
import matplotlib.pyplot as plt


class ReflectingBesselRandomWalk(MarkovChain):
    _delta = None

    def __init__(self, step_n, name, delta, random_seed=None):
        super().__init__(step_n, name, random_seed)
        self._delta = delta

    def _generate_path(self) -> np.ndarray:
        origin = 0
        positions = [origin]
        # Simulate steps
        _last_position = origin
        if self._random_seed:
            np.random.seed(self._random_seed)
        random_indexes = np.random.uniform(size=self._step_n)
        for j in range(0, self._step_n):
            # Calculate probability of getting a 1
            if _last_position == 0:
                # because the process is reflexive if we are at 0, the probability of going to 1 is 1
                _p_j = 1
            else:
                _p_j = np.divide(1, 2) * (
                    1 - np.divide(self._delta, 2 * _last_position)
                )
            # Obtain 1 or -1 depending on the probability
            uniform_val = random_indexes[j]
            if uniform_val <= _p_j:
                _var = 1
            else:
                _var = -1
            # Position for next step
            _last_position = _last_position + _var
            positions.append(_last_position)
        return np.array(positions)
