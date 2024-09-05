
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
        coefficients.append((signal * generate_wavelet_function (wavelet)).sum())
    
    #Normalize the coefficients
    coefficients = np.array(coefficients)
    coefficients = coefficients / np.sum(coefficients)
    
    def difference (coefficients, signal, base, difference_transformation = lambda x: x**2):
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
    wavelet_parts = [np.repeat (v, l) for v, l in wavelet[3:]]
    return np.concatenate (wavelet_parts)
 
def generate_function_from_wavelets (coefficients, base):
    """
    Generate a function from a set of wavelets.
    """
    assert base[0][0] == 0 & base[0][1] == 0; "The base is not ordered as expected."
    wf = np.zeros (base[0][3][1])
    for c, b in zip(coefficients, base):
        _,_,start, (va, na), (vb, nb) = b
        wf[start:start+na+nb] += (c * generate_wavelet_function (b))

    return np.array(wf)
                              
def test ():
    SA = np.sqrt (1/(250 - 0) - 1/(1000 - 0 + 1))
    SB = np.sqrt (1/(1000 - 250) - 1/(1000 - 0 + 1))
    base = [[0,0,  0, (10,1000),(0,0)],
            [1,0,  0, (SA,250), (SB, 750)]]

    coefficients = [1, 10]

    res = decompose (generate_function_from_wavelets (coefficients, base), base)
    return res.x - np.array (coefficients)