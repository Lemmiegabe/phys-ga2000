import numpy as np
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt

# Function to generate y values
def generate_y(N, num_samples):
    x = np.random.exponential(scale=1.0, size=(num_samples, N))
    y = np.mean(x, axis=1)
    return y

# Parameters
num_samples = 10000
N_values = np.arange(1, 901)

# Arrays to store results
means = []
variances = []
skewnesses = []
kurtoses = []

# Calculate mean, variance, skewness, and kurtosis for different N
for N in N_values:
    y = generate_y(N, num_samples)
    means.append(np.mean(y))
    variances.append(np.var(y))
    skewnesses.append(skew(y))
    kurtoses.append(kurtosis(y))
# Find the N where skewness and kurtosis are about 1% of their value for N=1
skewness_threshold = abs(skewnesses[0]) * 0.01
kurtosis_threshold = abs(kurtoses[0]) * 0.01

N_skewness = next(N for N, s in zip(N_values, skewnesses) if abs(s) < skewness_threshold)
N_kurtosis = next(N for N, k in zip(N_values, kurtoses) if abs(k) < kurtosis_threshold)

print(f'Skewness reaches 1% of its value for N=1 at N = {N_skewness}')
print(f'Kurtosis reaches 1% of its value for N=1 at N = {N_kurtosis}')
    

# Plotting
plt.figure(figsize=(12, 8))

# Mean plot
plt.subplot(2, 2, 1)
plt.plot(N_values, means, label='Mean')
plt.xlabel('N')
plt.ylabel('Mean')
plt.title('Mean vs N')
plt.grid(True)

# Variance plot
plt.subplot(2, 2, 2)
plt.plot(N_values, variances, label='Variance')
plt.xlabel('N')
plt.ylabel('Variance')
plt.title('Variance vs N')
plt.grid(True)

# Skewness plot
plt.subplot(2, 2, 3)
plt.plot(N_values, skewnesses, label='Skewness')
plt.xlabel('N')
plt.ylabel('Skewness')
plt.title('Skewness vs N')
plt.grid(True)

# Kurtosis plot
plt.subplot(2, 2, 4)
plt.plot(N_values, kurtoses, label='Kurtosis')
plt.xlabel('N')
plt.ylabel('Kurtosis')
plt.title('Kurtosis vs N')
plt.grid(True)

plt.tight_layout()
plt.savefig('central limit theorem.png')
plt.show()

