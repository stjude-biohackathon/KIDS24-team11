import numpy as np
import scipy.optimize as opt

# ------------------ Test Data Generation ------------------ #
def generate_test_data(cnv_string, noise):
    """
    Generate simple data for testing.
    """
    signal_values = np.concatenate([np.repeat(v, l) for v, l in cnv_string])
    noise_values = noise(np.sum([l for _, l in cnv_string]))
    return signal_values + noise_values

def noise(n, s=2):
    """
    Generate noise.
    """
    return np.random.normal(0, s, n)

def test_data():
    cnv_string = [(5, 100), (3, 1345), (4.9, 99), (3.1, 1345)]
    return generate_test_data(cnv_string, noise)

# ------------------ Haar Wavelet Functions ------------------ #
def haar_high(s, b, e):
    """
    This is the first term in the page 10 equation.
    """
    return np.sqrt(1 / (b - s + 1) - 1 / (e - s + 1))

def haar_low(s, b, e):
    """
    This is the second term in the page 10 equation.
    """
    return - np.sqrt(1 / (e - b) - 1 / (e - s + 1))

def basis_vector(s, b, e):
    """
    This creates a single basis vector for the haar wavelet, parameterized by s, b, e.
    """
    high = haar_high(s, b, e)
    low = haar_low(s, b, e)
    array = np.zeros(e - s)
    array[0:b - s] = high
    array[b - s:] = low
    return array

def haar_matrix(s, e):
    """
    This creates the matrix of basis vectors caused by iterating all possible break points.
    """
    matrix = np.zeros((e - s - 1, e - s))
    for i, b in enumerate(range(s + 1, e)):
        matrix[i, :] = basis_vector(s, b, e)
    return matrix

def choose_break(signal, s, e, p0=0.80, debug=False):
    """
    This function chooses the best break point for the signal between s and e.
    """
    if not (.5 <= p0 < 1):
        raise ValueError("p0 must be between [.5,1).")
    
    matrix = haar_matrix(s, e)
    scores = np.abs(np.matmul(matrix, signal[s:e]))
    
    trunc_scores = scores[int((1 - p0) * len(scores)):int(p0 * len(scores))]
    best_options = np.argwhere(trunc_scores == np.nanmax(trunc_scores)).flatten() + 1 + s + int((1 - p0) * len(scores))
    solution = best_options[np.abs(best_options - signal.size // 2).argmin()]
    
    if debug:
        return matrix, scores, best_options, solution
    else:
        return solution

def create_basis_form(s, b, e):
    """
    This function creates the basis form of the haar wavelet.
    """
    high = haar_high(s, b, e)
    low = haar_low(s, b, e)
    return [0, e - s, s, (high, b - s), (low, e - b)]

def generate_haar_basis(signal, p0=0.95, length=20, debug=False):
    """
    This function generates the haar basis for the signal.
    """
    d = length if isinstance(length, int) else int(length(signal.size))
    s = 0
    e = signal.size
    done = [[0, 0, 0, (signal.mean(), len(signal)), (0, 0)]]
    todo = [(0, s, e)]
    
    while len(todo) > 0:
        depth, s, e = todo.pop(0)
        if e - s >= d:
            if debug:
                print(f"todo: {len(todo)}, done: {len(done)}, len signal: {e-s}")
            break_point = choose_break(signal, s, e, p0)
            if break_point == s or break_point == e:
                break_point = (s + e) // 2
            solution = create_basis_form(s, break_point, e)
            solution[0] = depth
            if solution not in done:
                done.append(solution)
                todo.append((depth + 1, s, break_point))
                todo.append((depth + 1, break_point + 1, e))
    return done

# ------------------ Decomposition Functions ------------------ #
def decompose(signal, base):
    """
    Decompose the signal into a set of wavelets.
    """
    coefficients = []
    
    # Compute wavelet coefficients
    for wavelet in base:
        full_wavelet = np.zeros(len(signal))
        full_wavelet[wavelet[2]:wavelet[2] + wavelet[3][1] + wavelet[4][1]] = generate_wavelet_function(wavelet)
        coefficients.append((signal * full_wavelet).sum())
    
    # Normalize the coefficients
    coefficients = np.array(coefficients)
    coefficients = coefficients / np.sum(coefficients)

    # Define the difference function for optimization
    def difference(coefficients, signal, base, difference_transformation=lambda x: np.abs(x)):
        return np.sum(difference_transformation(signal - generate_function_from_wavelets(coefficients, base)))
    
    res = opt.minimize(difference, coefficients, args=(signal, base))
    return res

def generate_wavelet_function(wavelet):
    """
    Generate a wavelet function from a wavelet.
    """
    wavelet_parts = [np.repeat(v, l) for v, l in wavelet[3:]]
    return np.concatenate(wavelet_parts)

def generate_function_from_wavelets(coefficients, base):
    """
    Generate a function from a set of wavelets.
    """
    wf = np.zeros(base[0][3][1])
    for c, b in zip(coefficients, base):
        _, _, start, (_, na), (_, nb) = b
        wf[start:start + na + nb] += c * generate_wavelet_function(b)
    return wf

# ------------------ Running the Full Process ------------------ #
# Generate the test data (signal with noise)
signal = test_data()

# Generate the Haar wavelet basis for the signal
haar_basis = generate_haar_basis(signal, p0=0.95, length=20)

# Decompose the signal using the Haar wavelet basis
decomposition_result = decompose(signal, haar_basis)

# Print decomposition results
print("Decomposition coefficients:", decomposition_result.x)

# Reconstruct the signal from the decomposition coefficients
reconstructed_signal = generate_function_from_wavelets(decomposition_result.x, haar_basis)
reconstructed_signal

# Compare the original signal with the reconstructed signal
difference = signal - reconstructed_signal

# Print results
print("Original Signal:", signal)
print("Reconstructed Signal:", reconstructed_signal)
print("Difference between Original and Reconstructed Signal:", difference)

# Plotting the original signal, reconstructed signal, and their difference on the same plot
plt.figure(figsize=(12, 6))

plt.plot(signal, label='Original Signal', color='blue',alpha=0.7)
plt.plot(reconstructed_signal, label='Reconstructed Signal', color='green',alpha=0.3)
plt.plot(difference, label='Difference (Original - Reconstructed)', color='red',alpha=0.3)

plt.title('Original Signal, Reconstructed Signal, and Difference')
plt.legend()

# Display the plot
plt.show()
