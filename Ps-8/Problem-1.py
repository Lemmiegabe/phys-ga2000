from jax import grad, hessian
from jax.scipy.special import expit
from jax.scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Load the data using numpy
data = np.genfromtxt('survey.csv', delimiter=',', names=True, dtype=None, encoding=None)
ages = data['age']
responses = data['recognized_it']

# Standardize ages for better optimization performance
ages = (ages - np.mean(ages)) / np.std(ages)

# Define the logistic function
def logistic(x, beta0, beta1):
    return expit(beta0 + beta1 * x)

# Define the negative log likelihood function with debug print statements
def neg_log_likelihood(params):
    beta0, beta1 = params
    p = logistic(ages, beta0, beta1)
    likelihood = responses * jnp.log(p + 1e-10) + (1 - responses) * jnp.log(1 - p + 1e-10)
    neg_likelihood = -jnp.sum(likelihood)
    print(f"params: {params}, neg_log_likelihood: {neg_likelihood}")
    return neg_likelihood

# Initial guess for the parameters (not zeros)
initial_params = jnp.array([1.0, 0.1])

# Minimize the negative log likelihood with BFGS method and tighter tolerance
result = minimize(neg_log_likelihood, initial_params, method='BFGS', tol=1e-6)
beta0, beta1 = result.x

# Calculate the Hessian matrix and covariance matrix
hess = hessian(neg_log_likelihood)(result.x)
cov_matrix = jnp.linalg.inv(hess)

# Plot the results
plt.scatter(ages, responses,marker='.', label='Data')
age_range = jnp.linspace(min(ages), max(ages), 300)
predicted_prob = logistic(age_range, beta0, beta1)
plt.plot(age_range, predicted_prob, color='red', label='Logistic model')
plt.xlabel('Age (standardized)')
plt.ylabel('Probability of "Yes"')
plt.legend()
plt.savefig("/Dockerfile/image/output_plot.png")
print("Plot saved successfully.")


# Print the results
errors = jnp.sqrt(jnp.diag(cov_matrix))
print(f"Formal errors: β0 error = {errors[0]:.4f}, β1 error = {errors[1]:.4f}")
print(f"beta0: {beta0}, beta1: {beta1}")
print(f"Covariance matrix:\n{cov_matrix}")
