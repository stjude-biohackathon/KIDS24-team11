import numpy as np



def haar_high(s,b,e):
    """
    This is the first term in the page 10 equation.
    """
    return np.sqrt(1/(b - s + 1) - 1/(e - s + 1))

def haar_low(s,b,e):
    """
    This is the second term in the page 10 equation.
    """
    return - np.sqrt(1/(e - b) - 1/(e - s + 1))

def mother_haar(s,b,e,l):
    """
    Combine the terms to get the mother wavelet function.
    """
    term1 = haar_high(s,b,e) if s <= l <= b else 0
    term2 = haar_low(s,b,e) if b+1 <= l <= e else 0
    return term1 + term2

def basis_vector(s,b,e,l):
    """
    This creates a single basis vector for the haar wavelet, paramaterized by s,b,e and l.
    """
    high = haar_high(s,b,e)
    low = haar_low(s,b,e)
    array = np.zeros(l)
    array[b:e] = low
    array[s:b] = high
    return array

def haar_matrix(s,e,l):
    """
    This creates the matrix of basis vectors caused by iterating all possible break points.
    """
    matrix = np.zeros((e-s-1,l))
    for i, b in enumerate(range(s+1,e)):
        matrix[i] = basis_vector(s,b,e,l)
    return matrix

def choose_break(signal,s,e):
    """
    This function chooses the best break point for the signal between s and e.
    It resolves ties by selecting the break point closest to the center of the signal.
    """
    l = signal.size
    matrix = haar_matrix(s,e,l)
    scores = np.matmul(matrix,signal)
    best_options = np.argwhere(scores == np.nanmax(scores)).flatten() + s + 1
    return best_options[np.abs(best_options - l //2).argmin()]