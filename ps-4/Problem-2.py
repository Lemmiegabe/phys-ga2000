import numpy as np
import matplotlib.pyplot as plt

def potential(x):
    return x**4

def integrand(x, a):
    return 1 / np.sqrt(potential(a) - potential(x))

def period(a, N=20):
    m = 1  # mass of the particle
    # Get the Gaussian quadrature points and weights
    x, w = np.polynomial.legendre.leggauss(N)
    # Transform the points to the interval [0, a]
    x = 0.5 * a * (x + 1)
    w = 0.5 * a * w
    # Compute the integral using Gaussian quadrature
    integral = np.sum(w * integrand(x, a))
    T = 8 * np.sqrt(m) * integral
    return T

# Generate data for the graph
amplitudes = np.linspace(0.01, 2, 100)  # Avoid zero to prevent division by zero
periods = [period(a) for a in amplitudes]

# Plot the graph
plt.plot(amplitudes, periods)
plt.xlabel('Amplitude (a)')
plt.ylabel('Period (T)')
plt.title('Period of Anharmonic Oscillator vs Amplitude')
plt.grid(True)
#plt.savefig('period_vs_amplitude.png')
plt.show()