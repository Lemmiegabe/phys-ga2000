import numpy as np
from math import tanh

import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return 1 + 0.5 * tanh(2 * x)

# Define the analytic derivative of f(x)
def f_prime_analytic(x):
    return 1 / (np.cosh(2 * x) ** 2)

# Central difference method to calculate the numerical derivative
def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Range of x values
x_values = np.linspace(-2, 2, 400)
numerical_derivative = [central_difference(f, x) for x in x_values]
analytic_derivative = [f_prime_analytic(x) for x in x_values]

# Plotting the results
plt.plot(x_values, analytic_derivative, label='Analytic Derivative', color='blue')
plt.plot(x_values, numerical_derivative, '.', label='Numerical Derivative', color='red', markersize=2, alpha=0.4)
plt.xlabel('x')
plt.ylabel('Derivative')
plt.legend()
plt.title('Comparison of Numerical and Analytic Derivatives')
#plt.savefig('derivative_plot.png')
plt.show()