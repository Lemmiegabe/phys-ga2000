import numpy as np

# Get machine limits for 32-bit float
min_float32 = np.finfo(np.float32).min
max_float32 = np.finfo(np.float32).max
tiny_float32 = np.finfo(np.float32).tiny  # Smallest positive number
eps_float32 = np.finfo(np.float32).eps    # Precision

# Get machine limits for 64-bit float
min_float64 = np.finfo(np.float64).min
max_float64 = np.finfo(np.float64).max
tiny_float64 = np.finfo(np.float64).tiny  # Smallest positive number
eps_float64 = np.finfo(np.float64).eps    # Precision

# Print the results
print(f"32-bit float range: {min_float32} to {max_float32}")
print(f"32-bit smallest positive number: {tiny_float32}")
print(f"32-bit precision (epsilon): {eps_float32}")
print(f"epslion added to one: {1+eps_float32}")

print(f"64-bit float range: {min_float64} to {max_float64}")
print(f"64-bit smallest positive number: {tiny_float64}")
print(f"64-bit precision (epsilon): {eps_float64}")
print(f"epslion added to one: {1+eps_float64}")