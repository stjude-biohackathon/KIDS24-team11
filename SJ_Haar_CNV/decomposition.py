
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
        full_wavelet = np.zeros (len(signal))
        full_wavelet[wavelet[2]:wavelet[2]+wavelet[3][1]+wavelet[4][1]] = generate_wavelet_function (wavelet)
        coefficients.append((signal * full_wavelet).sum())
    
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
    
    wavelet_parts = [np.repeat (v, l) for v, l in wavelet[3:]]
    return np.concatenate (wavelet_parts)
 
def generate_function_from_wavelets (coefficients, base):
    """
    Generate a function from a set of wavelets.
    """
    wf = np.zeros (base[0][3][1])
    for c, b in zip(coefficients, base):
        _,_,start, (_, na),(_, nb) = b
        #print (c)
        #print (start,":", na, nb)
        #print (len(c*generate_wavelet_function (b)))
        wf[start:start+na+nb] = wf[start:start+na+nb] + (c * generate_wavelet_function (b))

    return wf
                              
def test_decompose ():
    SA = np.sqrt (1/(250 - 0) - 1/(1000 - 0 + 1))
    SB = np.sqrt (1/(1000 - 250) - 1/(1000 - 0 + 1))
    base = [[0,0,0, (10,1000),(0,0),(0,0)],
            [1,0,0, (SA,250), (SB, 750), (0,0)]]

    coefficients = [1, 10]

    res = decompose (generate_function_from_wavelets (coefficients, base), base)
    return res.x - np.array (coefficients)

def generate_output (coefficients, base):
    """
    Generate the output function from the truncated coefficients and the base.
    """
    
    pass