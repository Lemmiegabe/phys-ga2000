import numpy as np

def brent_method(f, a, b, tol=1e-5, max_iter=500):
    # Golden ratio
    golden_ratio = (3 - np.sqrt(5)) / 2

    # Initial points
    x = w = v = a + golden_ratio * (b - a)
    fx = fw = fv = f(x)
    
    d = e = b - a
    
    for iteration in range(max_iter):
        m = 0.5 * (a + b)
        tol1 = tol * abs(x) + tol / 10
        tol2 = 2 * tol1

        # Check for convergence
        if abs(x - m) <= (tol2 - 0.5 * (b - a)):
            return x, f(x), iteration

        # Parabolic fit
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)

            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                # Use parabolic interpolation
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = np.sign(m - x) * tol1
            else:
                # Use golden section
                e = d
                d = golden_ratio * (b - x if x < m else x - a)
        else:
            # Use golden section
            e = d
            d = golden_ratio * (b - x if x < m else x - a)

        u = x + d if abs(d) >= tol1 else x + np.sign(d) * tol1
        fu = f(u)

        # Update a, b, v, w, and x
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    # If max_iter reached without convergence
    raise ValueError("Brent's method did not converge")

# Define the test function
def test_function(x):
    return (x - 0.3)**2 * np.exp(x)

# Apply Brent's method
a, b = -1, 1  # Interval for minimization
x_min, f_min, iterations = brent_method(test_function, a, b)
brent_result = (x_min, f_min, iterations)
print(f"Brent's Method Result:\nMinimum x = {x_min}, f(x) = {f_min}, Iterations = {iterations}")

# Compare with scipy's brent method
from scipy.optimize import minimize_scalar
scipy_result = minimize_scalar(test_function, bounds=(a, b), method='bounded')
print(f"\nscipy.optimize.brent Result:\nMinimum x = {scipy_result.x}, f(x) = {scipy_result.fun}, Iterations = {scipy_result.nit}")

def print_difference(brent_result, scipy_result):
    x_diff = abs(brent_result[0] - scipy_result.x)
    f_diff = abs(brent_result[1] - scipy_result.fun)
    iter_diff = abs(brent_result[2] - scipy_result.nit)
    print(f"\nDifference between methods:\nDifference in x = {x_diff}\nDifference in f(x) = {f_diff}\nDifference in iterations = {iter_diff}")

# Print the difference
print_difference(brent_result, scipy_result)