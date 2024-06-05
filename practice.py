import numpy as np

# Define your binary vector
binary_vector = np.array([0, 1, 1, 0, 1, 0])

# Invert the binary vector using logical negation
inverted_vector = ~binary_vector.astype(bool)

print("Original binary vector:", binary_vector)
print("Inverted binary vector:", np.int32(inverted_vector))
