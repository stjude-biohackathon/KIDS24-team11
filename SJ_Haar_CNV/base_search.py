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

def basis_vector(s,b,e):
    """
    This creates a single basis vector for the haar wavelet, paramaterized by s,b,e and l.
    """
    high = haar_high(s,b,e)
    low = haar_low(s,b,e)
    array = np.zeros(e-s)
    array[0:b-s] = high
    array[b-s:] = low #should be b+1?
    return array

def haar_matrix(s,e):
    """
    This creates the matrix of basis vectors caused by iterating all possible break points.
    """
    matrix = np.zeros((e-s-1,e-s))
    for i, b in enumerate(range(s+1,e)):
        matrix[i,:] = basis_vector(s,b,e)
    return matrix

def choose_break(signal,s,e,p0 = .80, debug=False):
    """
    This function chooses the best break point for the signal between s and e.
    It resolves ties by selecting the break point closest to the center of the signal.
    """
    if not (.5 <= p0 < 1):
        raise ValueError("p0 must be between [.5,1).")
    offset = int((e-s)*(1-p0))
    matrix = haar_matrix(s,e)
    #scores = np.abs(np.matmul(matrix[offset:matrix.shape[0]-offset],signal[s:e]))
    scores = np.abs(np.matmul(matrix,signal[s:e]))
    trunc_scores = scores[int((1-p0)*len(scores)):int(p0*len(scores))]
    best_options = np.argwhere(trunc_scores == np.nanmax(trunc_scores)).flatten() + 1 + s + int((1-p0)*len(scores))
    solution = best_options[np.abs(best_options - signal.size //2).argmin()]
    if debug:
        return offset, matrix, scores, best_options, solution, matrix[solution - 1 - offset]
    else:
        return solution
    

def create_basis_form(s,b,e):
    """
    This function creates the basis form of the haar wavelet.
    """
    high = haar_high(s,b,e)
    low = haar_low(s,b,e)
    return [0,e-s, s,(high, b-s),(low,e-b)]

    
def generate_haar_basis(signal, p0 = .95, length = 20, debug=False):
    """
    This function generates the haar basis for the signal.
    """
    if callable(length):
        d = int(length(signal.size))
    elif isinstance(length, int):
        d = length
    else:
        raise ValueError("Depth must be an integer or a function.")
    
    if not (.5 <= p0 < 1):
        raise ValueError("p0 must be in [.5,1).")
    
    d = max(d,2)
    s = 0
    e = signal.size
    
    done = []
    todo = [(0, s,e)]
    while len(todo) > 0:
        depth, s,e = todo.pop(0)
        if e - s >= d:
            if debug:
                print(f"todo: {len(todo)}, done: {len(done)}, len signal: {e-s}")
            break_point = choose_break(signal,s,e,p0)
            if break_point == s or break_point == e:
                break_point = (s+e)//2
            solution = create_basis_form(s,break_point,e)
            solution[0] = depth
            if solution not in done:
                done.append(solution)
                todo.append((depth+1, s,break_point))
                todo.append((depth+1, break_point+1,e))
    return done

# def recursive_basis_generation(signal, s, e, d, p0=.95, debug=False):
#     """
#     This function handles the recursion of the basis generation.
#     """
    
#     if e - s >= d:
#         break_point = choose_break(signal,s,e,p0)
#         if debug: 
#             print(f"e: {e}, s: {s}, break_point: {break_point}")
#         return [create_basis_form(s,break_point,e)] + recursive_basis_generation(signal, s, break_point, d, p0) + recursive_basis_generation(signal, break_point, e, d, p0)
#     else:
#         return []
    