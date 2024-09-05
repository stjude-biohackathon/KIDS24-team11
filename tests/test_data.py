import numpy as np

def generate_test_data (cnv_string, noise):
    """
    Generate simple data for testing.
    """
    signal_values = np.concatenate([np.repeat (v, l) for v, l in cnv_string])
    noise_values = noise(np.sum([l for _, l in cnv_string]))
    return signal_values + noise_values, signal_values

def noise (n, s = 2):
    """
    Generate noise.
    """
    return np.random.normal(0, s, n)


def test_data ():
    cnv_string = [(5, 100), (3, 1345), (4.9, 99), (3.1, 1345)]
    return generate_test_data(cnv_string, noise)

