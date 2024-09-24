import numpy as np
import matplotlib.pyplot as plt
# Function to generate y values

num_samples = 10000
def generate_y(N, num_samples):
    x = np.random.exponential(scale=1.0, size=(num_samples, N))
    y = np.mean(x, axis=1)
    return y

# Function to plot histogram and Gaussian fit
def plot_histogram_with_gaussian(y, N):
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=50, density=True, alpha=0.6, color='g', label='Histogram')
    
    # Fit a normal distribution to the data
    mu, std = np.mean(y), np.std(y)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-((x - mu) ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
    
    plt.plot(x, p, 'k', linewidth=2, label='Gaussian fit')
    title = f'Central Limit Theorem '
    plt.title(title)
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.legend()
    plt.text(.95,1 , 'N = 900', fontsize=12)
    #plt.savefig(f'histogram_N_{N}_and_gaussian.png')
    plt.show()

# for large N
large_N = 900
y_large_N = generate_y(large_N, num_samples)
plot_histogram_with_gaussian(y_large_N, large_N)