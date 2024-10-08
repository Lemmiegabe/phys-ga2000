import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp

# Original user-defined function f(x)
def f(x):
    return 1 + 0.5 * math.tanh(2 * x)

# JAX version of the function
def f_jax(x):
    return 1 + 0.5 * jnp.tanh(2 * x)

# Analytical derivative of f(x)
def f_prime_analytic(x):
    return 1 / (math.cosh(2 * x) ** 2)

# Numerical derivative using central difference
def f_prime_numerical(x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# JAX automatic differentiation to find derivative
f_prime_autodiff = jax.grad(f_jax)

# Create an array of x values in the range -2 <= x <= 2
x_values = np.linspace(-2, 2, 400)

# Calculate derivatives
analytic_derivative = [f_prime_analytic(x) for x in x_values]
numerical_derivative = [f_prime_numerical(x) for x in x_values]
autodiff_derivative = [f_prime_autodiff(x) for x in x_values]

# Plotting the results
plt.plot(x_values, analytic_derivative, label='Analytic Derivative', linestyle='-', color='blue', alpha=0.5)
plt.plot(x_values, numerical_derivative, label='Numerical Derivative', linestyle='None', marker='o', markersize=2, color='red',alpha=0.3)
plt.plot(x_values, autodiff_derivative, label='Autodiff (JAX) Derivative', linestyle='--', color='green')

# Adding labels and legend
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Analytic vs Numerical vs Autodiff Derivative of f(x)")
plt.legend()
plt.grid()
# Save the plot to a file
#plt.savefig('derivative_plot.png')
plt.show()


