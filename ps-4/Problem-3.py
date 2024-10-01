import numpy as np
import matplotlib.pyplot as plt
import mpmath
from math import factorial, pi

# Define the Hermite polynomial using mpmath
def H(n, x):
    if n == 0:
        return mpmath.mpf(1)
    elif n == 1:
        return 2 * x
    else:
        # Use recurrence relation: H_n(x) = 2 * x * H_(n-1)(x) - 2 * (n-1) * H_(n-2)(x)
        H_nm2 = mpmath.mpf(1)
        H_nm1 = 2 * x
        for k in range(2, n + 1):
            H_n = 2 * x * H_nm1 - 2 * (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        return H_n

# Wavefunction using mpmath for higher precision
def psi(n, x):
    normalization = 1 / mpmath.sqrt(2**n * factorial(n) * mpmath.sqrt(pi))
    return normalization * mpmath.exp(-x**2 / 2) * H(n, x)

# Part c: Calculate the uncertainty for n = 5 using mpmath
evaluation_counter = 0  # Counter to track the number of evaluations

def integrand(x, n):
    global evaluation_counter
    evaluation_counter += 1
    return x**2 * abs(psi(n, x))**2

n = 5
# Calculate <x^2> using mpmath quadrature over an extended range for higher accuracy
result = mpmath.quad(lambda x: integrand(x, n), [-mpmath.inf, mpmath.inf])
uncertainty = mpmath.sqrt(result)

# Print the results
print(f'The uncertainty for n={n} using mpmath is approximately {float(uncertainty):.4f}')
print(f'Number of sample points used: {evaluation_counter}')

# Calculate the difference between 11/2 and the rms uncertainty
difference = np.sqrt(11 / 2) - float(uncertainty)
print(f'The difference between 11/2 and the rms uncertainty for n={n} is approximately {difference:.4f}')

# Part a: Plot the wavefunctions for n = 0, 1, 2, 3
x_vals = np.linspace(-4, 4, 400)
for n in range(4):
    # Convert mpmath output to numpy float for plotting
    psi_vals = [float(psi(n, x)) for x in x_vals]
    plt.plot(x_vals, psi_vals, label=f'n={n}')
plt.xlabel('x')
plt.ylabel(r'$\psi_n(x)$')
plt.title('Harmonic Oscillator Wavefunctions')
plt.legend()
plt.grid(True)
#plt.savefig('wavefunctions.png')
plt.show()

# Part b: Plot the wavefunction for n = 30
x_vals = np.linspace(-10, 10, 400)
psi_vals_30 = [float(psi(30, x)) for x in x_vals]
plt.plot(x_vals, psi_vals_30, label='n=30')
plt.xlabel('x')
plt.ylabel(r'$\psi_{30}(x)$')
plt.title('Harmonic Oscillator Wavefunction for n=30')
plt.legend()
plt.grid(True)
#plt.savefig('wavefunction_n30.png')
plt.show()
