import numpy as np


def upsample_time_series(data, target_length):
    """
    Upsample a time series to a target length using random sampling for normal distribution
    between consecutive points.
    
    """
    pass

def sample_from_normal_distribution(mean, std_dev, n_samples, seed=42):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.normal(loc=mean, scale=std_dev, size=n_samples)
    return samples





