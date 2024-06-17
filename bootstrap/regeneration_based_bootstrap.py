import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

# region Bootstrap algorithms


def single_regeneration_based_bootstrap(args: Tuple[np.ndarray, int]) -> List[float]:
    """
    Perform a single run of the regeneration-based bootstrap algorithm.

    Args:
        args (Tuple[np.ndarray, int]): A tuple containing:
            - block_values (np.ndarray): Array of block values.
            - num_blocks (int): The number of blocks.

    Returns:
        List[float]: A list of bootstrapped values.
    """
    block_values, num_blocks = args
    random_indices = np.random.randint(0, num_blocks, size=num_blocks)
    bootstrapped_values = [block_values[random_idx] for random_idx in random_indices]
    return bootstrapped_values


# endregion Bootstrap algorithms


def regeneration_based_bootstrap_series(
    block_values: List[float], num_bootstraps: int
) -> List[List[float]]:
    """
    Perform the regeneration-based bootstrap algorithm in series.

    Args:
        block_values (List[float]): List of block values.
        num_bootstraps (int): The number of bootstrap samples to generate.

    Returns:
        List[List[float]]: A list of lists, where each inner list contains bootstrapped values.
    """
    block_values = np.array(block_values)
    num_blocks = len(block_values)

    # Prepare the arguments for single_bootstrap_simulation
    args = [(block_values, num_blocks) for _ in range(num_bootstraps)]

    # Run the simulations in series
    bootstrap_results = [single_regeneration_based_bootstrap(arg) for arg in args]

    return bootstrap_results


# region parallelization


def regeneration_based_bootstrap_parallel_apply_async(
    block_values: List[float], num_bootstraps: int, num_processes: Optional[int] = None
) -> List[List[float]]:
    """
    Perform the regeneration-based bootstrap algorithm in parallel using apply_async.

    Args:
        block_values (List[float]): List of block values.
        num_bootstraps (int): The number of bootstrap samples to generate.
        num_processes (Optional[int]): The number of processes to use for parallelization. Defaults to the number of CPU cores.

    Returns:
        List[List[float]]: A list of lists, where each inner list contains bootstrapped values.
    """
    block_values = np.array(block_values)
    if not num_processes:
        num_processes = cpu_count()

    num_blocks = len(block_values)

    # Prepare the arguments for single_bootstrap_simulation
    args = [(block_values, num_blocks) for _ in range(num_bootstraps)]

    # Run the simulations in parallel using multiprocessing.Pool
    with Pool(processes=num_processes) as pool:
        async_results = [
            pool.apply_async(single_regeneration_based_bootstrap, (arg,))
            for arg in args
        ]
        bootstrap_results = [async_result.get() for async_result in async_results]

    return bootstrap_results


# endregion parallelization
