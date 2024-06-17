from markov_chain import ReflectingBesselRandomWalk, inverse_square, smaller_than
from bootstrap import (
    rbb_parallel_apply_async,
    regeneration_based_bootstrap_parallel_apply_async,
)
from utils import get_bootstrap_ci_mean, get_iid_ci_mean, get_coverage_probability
from experiments import get_coverage_probability_ssrw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def equal_to_one(block: np.array):
    return np.sum(block == 1)


def smaller_than(block: np.array):
    return np.sum(block < 10)


from multiprocessing import Pool


def _compute_mean_for_rep(args):
    n_steps, beta, delta, random_seed = args
    brw = ReflectingBesselRandomWalk(
        step_n=n_steps,
        name="Bessel random walk beta={0} delta={1}".format(beta, delta),
        delta=delta,
        random_seed=random_seed,
    )
    brw.generate_path()
    fn_blocks = brw.apply_fn_regeneration_blocks(state=0, fn=smaller_than)
    return np.mean(fn_blocks)


def obtain_mean_map(n_steps, beta, num_reps, random_seed):
    delta = 2 * beta - 1
    args = [
        (n_steps, beta, delta, random_seed + 10 * i) for i in range(1, num_reps + 1)
    ]

    with Pool() as pool:
        results = [pool.apply_async(_compute_mean_for_rep, (arg,)) for arg in args]
        means = [result.get() for result in results]

    return means


def obtain_mean_apply_async(n_steps, beta, num_reps, random_seed):
    delta = 2 * beta - 1
    args = [
        (n_steps, beta, delta, random_seed + 10 * i) for i in range(1, num_reps + 1)
    ]

    with Pool() as pool:
        means = pool.map(_compute_mean_for_rep, args)

    return means
