import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# 1. Graph of the Integrand (Part a)
def integrand(x, a):
    return x**(a - 1) * np.exp(-x)

def plot_integrands():
    # Values of 'a' for the different curves
    a_values = [2, 3, 4]
    # Define the range for x
    x = np.linspace(0, 5, 1000)

    # Plot the integrands
    plt.figure(figsize=(10, 6))
    for a in a_values:
        y = integrand(x, a)
        plt.plot(x, y, label=f'a = {a}')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graph of $\Gamma$ for a = 2, 3, 4')
    plt.legend()
    plt.grid()
   # plt.savefig('gamma_integrands.png')
    plt.show()

# 2. Finding Maximum Analytically (Part b)
def find_maximum():
    # Maximum occurs at x = a - 1
    print("Maximum occurs at x = a - 1")  # This is analytical, and we don't need implementation in code for demonstration

# 3. Change of Variables to Find 'c' (Part c)
def find_c(a):
    # Solving (a-1) / (c + (a-1)) = 1/2 for c
    return (a - 1)

# 4. Improved Expression for the Integrand (Part d)
def improved_integrand(x, a):
    # Using logarithms to avoid numerical overflow or underflow
    return np.exp((a - 1) * np.log(x) - x)

# 5. User-Defined Function gamma(a) (Part e)
def gamma(a):
    c = find_c(a)
    
    # Change of variable z = x / (c + x)
    def transformed_integrand(z):
        x = (c * z) / (1 - z)  # Inverse transformation
        return (c / (1 - z)**2) * improved_integrand(x, a)

    # Integrate over the range 0 to 1 for z
    result, _ = quad(transformed_integrand, 0, 1)
    return result

# 6. Verify Gamma(a) for Integers (Part f)
def verify_gamma_values():
    for a in [3, 6, 10]:
        gamma_value = gamma(a)
        factorial_value = math.factorial(a - 1)
        print(f"Γ({a}) ≈ {gamma_value}, Expected (factorial): {factorial_value}")

# Run the entire script sequentially
if __name__ == "__main__":
   
    # Part (b) - Maximum is analytically found to be at x = a - 1
    find_maximum()  # No code needed, just explanation

    # Part (c) - Find c for each a
    a_values = [2, 3, 4]
    c_values = [find_c(a) for a in a_values]
    print("Values of c for a = 2, 3, 4:", c_values)
    for a, c in zip(a_values, c_values):
        print(f"For a = {a}, c = {c}")

    # Part (e) - Test the gamma function for Γ(3/2)
    gamma_3_2 = gamma(1.5)
    print(" Γ(3/2) ≈", gamma_3_2)

    # Part (f) - Verify gamma function for integer values
    verify_gamma_values()
    # Part (a) - Plot the integrands
    plot_integrands()
