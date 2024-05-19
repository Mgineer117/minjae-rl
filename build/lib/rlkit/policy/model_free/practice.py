import numpy as np

# Create the x array (1000 x 10) filled with zeros
x = np.zeros((1000, 10))

# Example idx array (1000 x 1) with random indices between 0 and 9
idx = np.random.randint(0, 10, size=(1000, 1))

# Modify x according to idx
# We use np.arange to create an array of row indices
x[np.arange(1000), idx.flatten()] = 1

# Print some rows of x to verify
print(x[:10])
print(idx[:10])
print(idx.flatten()[:10])
