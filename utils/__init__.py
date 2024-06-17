import typing
from typing import List
import numpy as np

import scipy.stats as st


def get_normalized_mean_statistic(fn_blocks, mean, standard_dev=None):
    if not standard_dev:
        standard_dev = np.std(fn_blocks)
    return np.sqrt(len(fn_blocks)) * (np.mean(fn_blocks) - mean) / standard_dev


def get_bootstrap_ci(
    normalized_bootstrap_data: List[float],
    observed_mean: float,
    observed_standard_dev: float,
    observed_num_blocks: int,
    confidence_level: float = 0.95,
) -> np.array:
    alpha = 1 - confidence_level
    lower_quantile = np.quantile(normalized_bootstrap_data, q=1 - alpha / 2)
    upper_quantile = np.quantile(normalized_bootstrap_data, q=alpha / 2)

    ratio = observed_standard_dev / np.sqrt(observed_num_blocks)

    ci_lower = observed_mean - lower_quantile * ratio
    ci_upper = observed_mean - upper_quantile * ratio
    return np.array([ci_lower, ci_upper])


def get_bootstrap_ci_mean(
    bootstrap_data: List[List[float]],
    observed_mean: float,
    observed_standard_dev: float,
    num_blocks: int,
    confidence_level: float = 0.95,
) -> np.array:
    """
    Calculate the confidence interval for the mean of bootstrap data.

    Args:
        bootstrap_data (list[list[float]]): A list of lists containing bootstrap realizations.
        alpha (float, optional): The confidence level. Defaults to 0.95.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    normalized_bootstrap_data = list()
    for bootstrap_blocks in bootstrap_data:
        normalized_bootstrap_data.append(
            get_normalized_mean_statistic(
                fn_blocks=bootstrap_blocks,
                mean=observed_mean,
                standard_dev=observed_standard_dev,
            )
        )

    return get_bootstrap_ci(
        normalized_bootstrap_data=normalized_bootstrap_data,
        observed_mean=observed_mean,
        observed_standard_dev=observed_standard_dev,
        observed_num_blocks=num_blocks,
        confidence_level=confidence_level,
    )


def get_iid_ci_mean(data: List[float], confidence_level: float = 0.95) -> np.array:
    """
    Calculate the confidence interval for the mean of iid data.

    Args:
        data (list[float]): A list containing iid data points.
        alpha (float): The confidence level.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    return np.array(
        st.norm.interval(
            confidence=confidence_level, loc=np.mean(data), scale=st.sem(data)
        )
    )


def mean_ci_length(cis: np.ndarray) -> float:
    """
    Calculate the mean length of confidence intervals.

    Args:
        cis (np.ndarray): A 2D array where each row represents a confidence interval.
                          The first column is the lower bound and the second column is the upper bound.

    Returns:
        float: The mean length of the confidence intervals.
    """
    return np.mean(cis[:, 1] - cis[:, 0])


def get_coverage_probability(cis: np.ndarray, true_value: float) -> float:
    """
    Calculate the coverage probability of the true value within the given confidence intervals.

    Args:
        cis (np.ndarray): A 2D array where each row represents a confidence interval.
                          The first column is the lower bound and the second column is the upper bound.
        true_value (float): The true value to check against the confidence intervals.

    Returns:
        float: The coverage probability, i.e., the proportion of confidence intervals that contain the true value.
    """
    count = np.sum((cis[:, 0] <= true_value) & (cis[:, 1] >= true_value))
    return count / cis.shape[0]
