import typing
from markov_chain import SimpleSymmetricRandomWalk
from bootstrap import (
    rbb_series,
    regeneration_based_bootstrap_series,
)
from utils import (
    get_bootstrap_ci_mean,
    get_iid_ci_mean,
    get_coverage_probability,
    mean_ci_length,
)
import numpy as np


import multiprocessing


def process_chain(args):
    i, n_steps, fn, n_bootstraps, confidence_level, random_seed = args
    while True:
        # print(i)
        random_seed += 1
        # Generate the chain
        ssrw = SimpleSymmetricRandomWalk(
            step_n=n_steps, name=f"simple_rw_{i}", random_seed=random_seed
        )
        ssrw.generate_path()
        # Get regeneration block sizes
        regeneration_block_sizes = ssrw.get_regeneration_block_sizes(state=0)
        # Get the value of the function evaluated on the blocks
        fn_blocks = ssrw.apply_fn_regeneration_blocks(state=0, fn=fn)
        if np.sum(regeneration_block_sizes) == 0 or np.std(fn_blocks) == 0:
            continue
        # Apply RBB
        fn_rbb_bootstraps = rbb_series(
            block_sizes=regeneration_block_sizes,
            block_values=fn_blocks,
            n=n_steps,
            num_bootstraps=n_bootstraps,
        )

        # Apply regeneration based bootstrap
        fn_regeneration_based_bootstraps = regeneration_based_bootstrap_series(
            block_values=fn_blocks, num_bootstraps=n_bootstraps
        )

        # Get confidence intervals
        rbb_ci = get_bootstrap_ci_mean(
            bootstrap_data=fn_rbb_bootstraps,
            observed_mean=np.mean(fn_blocks),
            observed_standard_dev=np.std(fn_blocks),
            num_blocks=len(fn_blocks),
            confidence_level=confidence_level,
        )
        regeneration_based_ci = get_bootstrap_ci_mean(
            bootstrap_data=fn_regeneration_based_bootstraps,
            observed_mean=np.mean(fn_blocks),
            observed_standard_dev=np.std(fn_blocks),
            num_blocks=len(fn_blocks),
            confidence_level=confidence_level,
        )
        iid_ci = get_iid_ci_mean(data=fn_blocks, confidence_level=confidence_level)
        return rbb_ci, regeneration_based_ci, iid_ci


def get_confidence_intervals_ssrw_async(
    n_steps: int,
    fn: typing.Callable,
    n_reps: int,
    n_bootstraps: int,
    confidence_level: float,
    random_seed: int,
):
    rbb_cis = np.empty((n_reps, 2))
    regeneration_based_cis = np.empty((n_reps, 2))
    iid_cis = np.empty((n_reps, 2))

    with multiprocessing.Pool() as pool:
        results = [
            pool.apply_async(
                process_chain,
                [
                    (
                        i,
                        n_steps,
                        fn,
                        n_bootstraps,
                        confidence_level,
                        random_seed + i * 10,
                    )
                ],
            )
            for i in range(n_reps)
        ]
        for i, result in enumerate(results):
            rbb_cis[i], regeneration_based_cis[i], iid_cis[i] = result.get()

    return rbb_cis, regeneration_based_cis, iid_cis


def get_coverage_probability_ssrw(
    n_steps: int,
    fn: typing.Callable[[np.ndarray], float],
    n_reps: int,
    n_bootstraps: int,
    true_mean: float,
    confidence_level: float,
    random_seed: int,
) -> typing.Tuple[
    typing.Dict[str, typing.Any],
    typing.Dict[str, typing.Any],
    typing.Dict[str, typing.Any],
]:
    """
    Calculate the coverage probability and average length of confidence intervals for a simple symmetric random walk.

    Parameters:
    n_steps (int): Number of steps in the random walk.
    fn (typing.Callable[[np.ndarray], float]): Function to apply to the regeneration blocks.
    n_reps (int): Number of repetitions for the experiment.
    n_bootstraps (int): Number of bootstrap samples.
    true_mean (float): The true mean value to compare against.
    random_seed (int): Seed for random number generation.

    Returns:
    typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]:
        A tuple containing dictionaries with confidence intervals, coverage probabilities, and average lengths for:
        - Regeneration block bootstrap (RBB)
        - Regeneration based bootstrap
        - IID bootstrap
    """
    rbb_cis, regeneration_based_cis, iid_cis = get_confidence_intervals_ssrw_async(
        n_steps=n_steps,
        fn=fn,
        n_reps=n_reps,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    rbb_coverage_prob = get_coverage_probability(rbb_cis, true_value=true_mean)
    regeneration_based_coverage_prob = get_coverage_probability(
        regeneration_based_cis, true_value=true_mean
    )
    iid_coverage_prob = get_coverage_probability(iid_cis, true_value=true_mean)

    rbb_ci_avg_length = mean_ci_length(cis=rbb_cis)
    regeneration_based_ci_avg_length = mean_ci_length(cis=regeneration_based_cis)
    iid_ci_avg_length = mean_ci_length(cis=iid_cis)

    print(
        f"The coverage probability for the RBB is {rbb_coverage_prob}. The average length is {rbb_ci_avg_length}"
    )
    print(
        f"The coverage probability for the regeneration based bootstrap is {regeneration_based_coverage_prob}. The average length is {regeneration_based_ci_avg_length}"
    )
    print(
        f"The coverage probability using the IID data is {iid_coverage_prob}. The average length is {iid_ci_avg_length}"
    )

    return (
        {
            "c_i": rbb_cis,
            "cov_prob": rbb_coverage_prob,
            "avg_length": rbb_ci_avg_length,
        },
        {
            "c_i": regeneration_based_cis,
            "cov_prob": regeneration_based_coverage_prob,
            "avg_length": regeneration_based_ci_avg_length,
        },
        {
            "c_i": iid_cis,
            "cov_prob": iid_coverage_prob,
            "avg_length": iid_ci_avg_length,
        },
    )
