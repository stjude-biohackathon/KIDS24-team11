
import numpy as np
import scipy.optimize as opt


def decompose (signal, base):
    """
    Decompose the signal into a set of wavelets.
    """
    
    coefficients = []
    # Initial set of coefficient
    for wavelet in base:
        # Compute the wavelet coefficients.
        coefficients.append = signal * generate_wavelet_function (wavelet)
    
    #Normalize the coefficients
    coefficients = np.array(coefficients)
    coefficients = coefficients / np.sum(coefficients)

    def difference (coefficients, signal, base, difference_transformation = lambda x: np.abs(x)):
        """
        Compute the difference between the signal and the sum of wavelets.
        """
        return np.sum (difference_transformation(signal - generate_function_from_wavelets (coefficients, base)))
    
    res = opt.minimize (difference, coefficients, args=(signal, base))

    return res

def generate_wavelet_function (wavelet):
    """
    Generate a wavelet function from a wavelet.
    """
    wavelet_parts = [np.repeat (v, l) for v, l in wavelet[2:]]
    return np.concatenate (wavelet_parts)
 
def generate_function_from_wavelets (coefficients, base):
    """
    Generate a function from a set of wavelets.
    """
    wf = []
    for c, b in zip(coefficients, base):
        wf.append (c * generate_wavelet_function (b))

    return np.sum(np.array(wf).reshape (len(wf[0]), len(base)), axis=1)
                              

