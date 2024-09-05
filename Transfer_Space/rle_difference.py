import matplotlib.pyplot as plt

# Decode RLE sequence into a regular sequence
def decode_rle(rle):
    decoded = []
    for value, count in rle:
        decoded.extend([value] * count)
    return decoded

# Encode a sequence into RLE format
def encode_rle(sequence):
    if not sequence:
        return []
    rle = []
    current_value = sequence[0]
    current_count = 1
    for value in sequence[1:]:
        if value == current_value:
            current_count += 1
        else:
            rle.append((current_value, current_count))
            current_value = value
            current_count = 1
    rle.append((current_value, current_count))
    return rle

# Compare two RLEs and return the RLE of their differences
def compare_rle_as_vectors(rle1, rle2):
    decoded1 = decode_rle(rle1)
    decoded2 = decode_rle(rle2)
    
    max_len = max(len(decoded1), len(decoded2))
    decoded1 += [0] * (max_len - len(decoded1))
    decoded2 += [0] * (max_len - len(decoded2))
    
    differences = [a - b for a, b in zip(decoded1, decoded2)]
    return encode_rle(differences)

# Plot the RLE comparison and highlight differences with horizontal red lines
def plot_rle_comparison(rle1, rle2, rle_diff):
    decoded1 = decode_rle(rle1)
    decoded2 = decode_rle(rle2)
    decoded_diff = decode_rle(rle_diff)
    
    # Create the x-axis positions for each value in the decoded sequences
    x1 = list(range(len(decoded1)))
    x2 = list(range(len(decoded2)))
    
    plt.figure(figsize=(12, 6))
    
    # Plot RLE 1 and RLE 2 as step plots
    plt.step(x1, decoded1, where='mid', label='RLE 1', alpha=0.7)
    plt.step(x2, decoded2, where='mid', label='RLE 2', alpha=0.7)
    
    # Highlight the differences with horizontal red lines across ranges
    start_diff = None
    for i in range(len(decoded_diff)):
        if decoded_diff[i] != 0 and start_diff is None:
            start_diff = i  # Start of a difference
        elif decoded_diff[i] == 0 and start_diff is not None:
            # End of a difference range, plot a horizontal line
            plt.hlines(y=decoded1[start_diff], xmin=start_diff, xmax=i, color='red', lw=3)
            start_diff = None
    
    # If there's a difference at the very end
    if start_diff is not None:
        plt.hlines(y=decoded1[start_diff], xmin=start_diff, xmax=len(decoded_diff), color='red', lw=3)

    plt.legend()
    plt.title('Comparison of Two RLE Sequences with Differences Highlighted')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.show()

# Example usage
rle1 = [(1, 3), (2, 4), (3, 10), (1, 1)]
rle2 = [(1, 3), (2, 3), (3, 3), (2, 5)]

rle_diff = compare_rle_as_vectors(rle1, rle2)
print("RLE of differences:")
print(rle_diff)

# Plot the comparison between the two RLE sequences and highlight the differences
plot_rle_comparison(rle1, rle2, rle_diff)
