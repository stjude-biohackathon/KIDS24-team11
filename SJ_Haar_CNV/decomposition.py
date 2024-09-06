
import numpy as np
import scipy.optimize as opt


def haar_basis(l,base):
    res = np.zeros((len(base),l))
    for i, wavelet in enumerate(base):
        res[i,wavelet[2]:wavelet[2]+wavelet[3][1]+wavelet[4][1]] = generate_wavelet_function(wavelet)
    return res
    

def decompose (signal, base, threshold = None, k = 1.4826):
    """
    Decompose the signal into a set of wavelets.
    """
    
    
    if threshold is None:
        threshold = estimate_threshold (signal, k)
    
    basis = haar_basis(len(signal),base)
    coefficients = np.matmul(basis,signal)
    #normalized_coefficients = coefficients / coefficients.sum()
    drop = np.argwhere(np.abs(coefficients) < threshold).flatten()
    coefficients[0] = 1.0
    all_coefficients = coefficients.copy()
    coefficients[drop] = 0.0
    y = generate_function_from_wavelets (coefficients, base,len(signal))
    
    return basis, coefficients, all_coefficients, threshold, rle(y)

def rle(s):
    counts = {}
    for c in s:
        if counts.get(c) is None:
            counts[c] = 1
        else:
            counts[c] = counts[c] + 1
    return [(k,counts[k]) for k in counts.keys()]



def estimate_threshold (X, k = 1.4826):
    d = np.abs(X[1:]-X[:-1])/np.sqrt(2)
    MAD = np.median(d)
    sigma = k*MAD
    thr = sigma*np.sqrt(2*np.log(len(X)))
    return thr

def generate_wavelet_function (wavelet):
    """
    Generate a wavelet function from a wavelet.
    """
    
    wavelet_parts = [np.repeat (v, l) for v, l in wavelet[3:]]
    return np.concatenate (wavelet_parts)
 
def generate_function_from_wavelets (coefficients, base, l):
    """
    Generate a function from a set of wavelets.
    """
    #wf = np.zeros (base[0][3][1])
    #for c, b in zip(coefficients, base):
    #    _,_,start, (_, na),(_, nb) = b
    #    wf[start:start+na+nb] = wf[start:start+na+nb] + (c * generate_wavelet_function (b))
    wf = haar_basis (l, base) 
    return (wf* coefficients[ :, np.newaxis]).sum(axis = 0)
                              
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

def rle_to_array(rle):
    return np.concatenate([np.repeat(k,v) for k,v in rle])

# import numpy as np
# import scipy.optimize as opt


# def decompose (signal, base):
#     """
#     Decompose the signal into a set of wavelets.
#     """
    
#     coefficients = []
#     # Initial set of coefficient
    
#     for wavelet in base:
#         # Compute the wavelet coefficients.
#         full_wavelet = np.zeros (len(signal))
#         full_wavelet[wavelet[2]:wavelet[2]+wavelet[3][1]+wavelet[4][1]] = generate_wavelet_function (wavelet)
#         coefficients.append((signal * full_wavelet).sum())
    
#     #Normalize the coefficients
    
#     coefficients = np.array(coefficients)
#     coefficients = coefficients / np.sum(coefficients)

#     def difference (coefficients, signal, base, difference_transformation = lambda x: np.abs(x)):
#         """
#         Compute the difference between the signal and the sum of wavelets.
#         """
#         return np.sum (difference_transformation(signal - generate_function_from_wavelets (coefficients, base)))
    
#     res = opt.minimize (difference, coefficients, args=(signal, base))

#     return res

# def generate_wavelet_function (wavelet):
#     """
#     Generate a wavelet function from a wavelet.
#     """
    
#     wavelet_parts = [np.repeat (v, l) for v, l in wavelet[3:]]
#     return np.concatenate (wavelet_parts)
 
# def generate_function_from_wavelets (coefficients, base):
#     """
#     Generate a function from a set of wavelets.
#     """
#     wf = np.zeros (base[0][3][1])
#     for c, b in zip(coefficients, base):
#         _,_,start, (_, na),(_, nb) = b
#         #print (c)
#         #print (start,":", na, nb)
#         #print (len(c*generate_wavelet_function (b)))
#         wf[start:start+na+nb] = wf[start:start+na+nb] + (c * generate_wavelet_function (b))

#     return wf
                              
# def test_decompose ():
#     SA = np.sqrt (1/(250 - 0) - 1/(1000 - 0 + 1))
#     SB = np.sqrt (1/(1000 - 250) - 1/(1000 - 0 + 1))
#     base = [[0,0,0, (10,1000),(0,0),(0,0)],
#             [1,0,0, (SA,250), (SB, 750), (0,0)]]

#     coefficients = [1, 10]

#     res = decompose (generate_function_from_wavelets (coefficients, base), base)
#     return res.x - np.array (coefficients)

# def generate_output (coefficients, base):
#     """
#     Generate the output function from the truncated coefficients and the base.
#     """
    
#     pass