import numpy as np
from datetime import datetime, time

def block_bootstrap(data, block_size=20, num_samples=100):
    """
    Perform block bootstrap sampling

    Args:
        data: Data list or array
        block_size: Size of each block
        num_samples: Number of samples to generate

    Returns:
        list: List of bootstrap samples
    """
    data = np.array(data)
    n = len(data)
    samples = []

    for _ in range(num_samples):
        # Select random starting points
        num_blocks = int(np.ceil(n / block_size))
        # Maximum n - block_size + 1 starting points possible
        max_start = max(1, n - block_size + 1)
        start_indices = np.random.randint(0, max_start, num_blocks)

        # Combine blocks
        bootstrap_sample = []
        for start in start_indices:
            end = min(start + block_size, n)
            bootstrap_sample.extend(data[start:end])

            # Exit loop if enough data is collected
            if len(bootstrap_sample) >= n:
                bootstrap_sample = bootstrap_sample[:n]
                break

        samples.append(bootstrap_sample)

    return samples

def weighted_bootstrap(data, weights=None, block_size=20, num_samples=100):
    """
    Perform weighted bootstrap sampling

    Args:
        data: Data list or array
        weights: Weights for data points (None = last 1000 60%, rest 40%)
        block_size: Size of each block
        num_samples: Number of samples to generate

    Returns:
        list: List of bootstrap samples
    """
    data = np.array(data)
    n = len(data)

    # Define weights (if not given, last 1000 60%, rest 40%)
    if weights is None:
        weights = np.ones(n)
        if n > 1000:
            weights[-1000:] = 1.5  # 50% more weight to last 1000 samples

    # Normalize weights
    weights = weights / np.sum(weights)

    samples = []
    for _ in range(num_samples):
        # Select block starting points with weights
        num_blocks = int(np.ceil(n / block_size))
        sample_indices = []

        for _ in range(num_blocks):
            # Weighted sampling
            start = np.random.choice(range(n), p=weights)

            # Add block
            end = min(start + block_size, n)
            block = list(range(start, end))
            sample_indices.extend(block)

            # Exit loop if enough data is collected
            if len(sample_indices) >= n:
                sample_indices = sample_indices[:n]
                break

        # Make indices unique and sort
        sample_indices = sorted(list(set(sample_indices)))

        # Sample according to indices
        bootstrap_sample = data[sample_indices]
        samples.append(bootstrap_sample)

    return samples

def time_based_bootstrap(data, timestamps=None, block_size=20, num_samples=100):
    """
    Perform time-based bootstrap sampling, but uses normal block bootstrap if no time data

    Args:
        data: Data list or array
        timestamps: Time stamps (not used)
        block_size: Size of each block
        num_samples: Number of samples to generate

    Returns:
        list: List of bootstrap samples
    """
    # Use normal block bootstrap since no time data
    return block_bootstrap(data, block_size, num_samples)
