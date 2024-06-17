from typing import List, Tuple, Optional
import numpy as np
from multiprocessing import Pool, cpu_count


def single_rbb(args: Tuple[np.ndarray, np.ndarray, int, int]) -> List[float]:
    """
    Perform a single run of the regeneration-based bootstrap (RBB) algorithm.

    Args:
        args (Tuple[np.ndarray, np.ndarray, int, int]): A tuple containing:
            - block_sizes (np.ndarray): Array of block sizes.
            - block_values (np.ndarray): Array of block values.
            - n (int): The target sum for the bootstrap.
            - batch_size (int): The size of the batch for random index generation.

    Returns:
        List[float]: A list of bootstrapped values.
    """
    block_sizes, block_values, n, batch_size = args
    current_sum = 0
    bootstrapped_values = []
    random_indices = np.random.randint(0, len(block_sizes), size=batch_size)
    index_counter = 0

    while True:
        # Get a random index from the pre-generated indices
        random_idx = random_indices[index_counter]

        # Add the block size to the current sum
        current_sum += block_sizes[random_idx]

        # If the sum exceeds n, we break out of the loop
        if current_sum > n:
            break

        # Add the bootstrapped value
        bootstrapped_values.append(block_values[random_idx])

        # Update the index_counter and generate more indexes if necessary
        index_counter += 1
        if index_counter >= batch_size:
            random_indices = np.random.randint(0, len(block_sizes), size=batch_size)
            index_counter = 0

    return bootstrapped_values


def rbb_series(
    block_sizes: List[float],
    block_values: List[float],
    n: int,
    num_bootstraps: int,
) -> List[List[float]]:
    """
    Perform the regeneration-based bootstrap (RBB) algorithm in series.

    Args:
        block_sizes (List[float]): List of block sizes.
        block_values (List[float]): List of block values.
        n (int): The target sum for the bootstrap.
        num_bootstraps (int): The number of bootstrap samples to generate.

    Returns:
        List[List[float]]: A list of lists, where each inner list contains bootstrapped values.
    """
    block_sizes = np.array(block_sizes)
    block_values = np.array(block_values)

    # Calculate the batch_size based on n and block_sizes
    batch_size = int(n / np.sum(block_sizes)) * len(block_sizes) + 1

    # Prepare the arguments for single_bootstrap_simulation
    args = [(block_sizes, block_values, n, batch_size) for _ in range(num_bootstraps)]

    # Run the simulations in series
    bootstrap_results = [single_rbb(arg) for arg in args]

    return bootstrap_results


def rbb_parallel_apply_async(
    block_sizes: List[float],
    block_values: List[float],
    n: int,
    num_bootstraps: int,
    num_processes: Optional[int] = None,
) -> List[List[float]]:
    """
    Perform the regeneration-based bootstrap (RBB) algorithm in parallel using apply_async.

    Args:
        block_sizes (List[float]): List of block sizes.
        block_values (List[float]): List of block values.
        n (int): The target sum for the bootstrap.
        num_bootstraps (int): The number of bootstrap samples to generate.
        num_processes (Optional[int]): The number of processes to use for parallelization. Defaults to the number of CPU cores.

    Returns:
        List[List[float]]: A list of lists, where each inner list contains bootstrapped values.
    """
    block_sizes = np.array(block_sizes)
    block_values = np.array(block_values)
    if not num_processes:
        num_processes = cpu_count()

    # Calculate the batch_size based on n and block_sizes
    batch_size = int(n / np.sum(block_sizes)) * len(block_sizes) + 1

    # Prepare the arguments for single_bootstrap_simulation
    args = [(block_sizes, block_values, n, batch_size) for _ in range(num_bootstraps)]

    # Run the simulations in parallel using multiprocessing.Pool
    with Pool(processes=num_processes) as pool:
        async_results = [pool.apply_async(single_rbb, (arg,)) for arg in args]
        bootstrap_results = [async_result.get() for async_result in async_results]

    return bootstrap_results
