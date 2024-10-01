import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

# Constants
V = 1000 * 1e-6  # Volume in cubic meters
rho = 6.022e28  # Number density in m^-3
k_B = 1.380649e-23  # Boltzmann constant in J/K
theta_D = 428  # Debye temperature in K

def integrand(x):
    return (x**4 * np.exp(x)) / (np.exp(x) - 1)**2

def cv(T, N=50):
    x, w = leggauss(N)
    x = 0.5 * (x + 1) * (theta_D / T)  # Transform the interval from [-1, 1] to [0, theta_D/T]
    w = 0.5 * w * (theta_D / T)  # Adjust weights accordingly
    integral = np.sum(w * integrand(x))
    return 9 * V * rho * k_B * (T / theta_D)**3 * integral
# Part c: Test convergence with different N values
N_values = [10, 30, 40, 50, 60, 70]
T_test = 300  # Test temperature

for N in N_values:
    print(f'N = {N}, C_V = {cv(T_test, N)}')

# Part b: Graph of heat capacity as a function of temperature
temperatures = np.linspace(5, 500, 100)
heat_capacities = [cv(T) for T in temperatures]

plt.plot(temperatures, heat_capacities)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.title('Heat Capacity of Solid Aluminum as a Function of Temperature')
plt.grid(True)
#plt.savefig('heat_capacity.png')
plt.show()


