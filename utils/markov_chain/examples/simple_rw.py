import math
import os
import numpy as np
from utils.markov_chain import MarkovChain
import matplotlib.pyplot as plt


class SimpleSymmetricRandomWalk(MarkovChain):

    def __init__(self, step_n, name, p=np.divide(1, 2)):
        super().__init__(step_n, name)

    def _generate_path(self) -> np.ndarray:
        step_set = [-1, 1]
        origin = np.zeros((1, 1))
        # Simulate steps
        step_shape = (self._step_n, 1)
        steps = np.random.choice(a=step_set, size=step_shape)
        return np.concatenate([origin, steps]).cumsum(0)
