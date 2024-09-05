import numpy as np
from scipy import stats
from typing import Tuple, List

def generate_diploid_signal(length: int, noise_std: float = 0.1) -> np.ndarray:
    """
    Generate a diploid signal (copy number 2) with added noise.
    
    Args:
    length (int): Length of the signal.
    noise_std (float): Standard deviation of the noise.
    
    Returns:
    np.ndarray: Diploid signal with noise.
    """
    base_signal = np.full(length, 2.0)  # Diploid base (copy number 2)
    noise = np.random.normal(0, noise_std, length)
    return base_signal + noise

def generate_signal_with_deletion(length: int, deletion_start: int, deletion_end: int, noise_std: float = 0.1) -> np.ndarray:
    """
    Generate a diploid signal with a deletion and added noise.
    
    Args:
    length (int): Length of the signal.
    deletion_start (int): Start position of the deletion.
    deletion_end (int): End position of the deletion.
    noise_std (float): Standard deviation of the noise.
    
    Returns:
    np.ndarray: Signal with deletion and noise.
    """
    signal = np.full(length, 2.0)  # Diploid base (copy number 2)
    signal[deletion_start:deletion_end] = 1.0  # Deletion (copy number 1)
    noise = np.random.normal(0, noise_std, length)
    return signal + noise

def segment_signal(signal: np.ndarray, window_size: int = 100) -> List[Tuple[float, float]]:
    """
    Segment the signal into windows and calculate mean and standard deviation.
    
    Args:
    signal (np.ndarray): Input signal.
    window_size (int): Size of the sliding window.
    
    Returns:
    List[Tuple[float, float]]: List of (mean, std) tuples for each window.
    """
    return [
        (np.mean(signal[i:i+window_size]), np.std(signal[i:i+window_size]))
        for i in range(0, len(signal) - window_size + 1, window_size)
    ]

def test_case_1(signal: np.ndarray, window_size: int = 100, noise_std: float = 0.1) -> bool:
    """
    Comprehensive test case for diploid signal analysis.
    
    Args:
    signal (np.ndarray): Input signal to test.
    window_size (int): Size of the sliding window for segmentation.
    noise_std (float): Standard deviation of the noise used in signal generation.
    
    Returns:
    bool: True if all tests pass, False otherwise.
    """
    segments = segment_signal(signal, window_size)
    means, stds = zip(*segments)
    overall_mean = np.mean(means)
    mean_std = np.std(means)
    expected_std = noise_std / np.sqrt(window_size)
    t_statistic, p_value = stats.ttest_1samp(means, 2.0)
    
    print(f"Overall mean of segments: {overall_mean:.4f}")
    print(f"Standard deviation of segment means: {mean_std:.4f}")
    print(f"T-test p-value: {p_value:.4f}")
    
    # Test 1: Check if the mean of segment means is close to 2
    test1 = 1.95 < overall_mean < 2.05
    print(f"Test 1 {'Passed' if test1 else 'Failed'}: Overall mean is {'close to' if test1 else 'not close to'} 2")
    
    # Test 2: Check if the standard deviation of means is within expected range
    test2 = expected_std * 0.8 < mean_std < expected_std * 1.2
    print(f"Test 2 {'Passed' if test2 else 'Failed'}: Standard deviation of means is {'within' if test2 else 'outside'} expected range")
    
    # Test 3: Check if the mean is not significantly different from 2
    test3 = p_value > 0.05
    print(f"Test 3 {'Passed' if test3 else 'Failed'}: Mean is {'not' if test3 else ''} significantly different from 2")
    
    return all([test1, test2, test3])

# Test with diploid data
print("Testing with diploid data:")
diploid_signal = generate_diploid_signal(10000, 0.1)
diploid_result = test_case_1(diploid_signal)
print(f"Overall test {'Passed' if diploid_result else 'Failed'} for diploid data\n")

# Test with data containing a deletion
print("Testing with data containing a deletion:")
deletion_signal = generate_signal_with_deletion(10000, 3000, 7000, 0.1)
deletion_result = test_case_1(deletion_signal)
print(f"Overall test {'Passed' if deletion_result else 'Failed'} for data with deletion")
