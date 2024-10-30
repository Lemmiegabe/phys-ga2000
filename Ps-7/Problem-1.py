from jax import grad


def f_rescaled_jax(r_prime, m_prime):
    """Rescaled force balance function in terms of r' and m'."""
    return 1 / (r_prime ** 2) - m_prime / ((1 - r_prime) ** 2) - (1 + m_prime) * r_prime

def find_L1_distance_jax(M, m, R, initial_guess, system_name, tolerance=1e-10, max_iterations=1000):
    """Calculate the L1 distance using JAX's automatic differentiation."""
    m_prime = m / M  # Mass ratio for rescaling
    print(f"\n{system_name} system:")
    print(f"Mass ratio (m'/M): {m_prime:.3e}")

    # JAX's grad function to calculate derivative of f_rescaled with respect to r_prime
    df_rescaled_jax = grad(f_rescaled_jax, argnums=0)
    
    # Newton's method loop using JAX
    r_prime = initial_guess
    for iteration in range(max_iterations):
        f_val = f_rescaled_jax(r_prime, m_prime)
        df_val = df_rescaled_jax(r_prime, m_prime)
        
        # Check for zero derivative to prevent division by zero
        if abs(df_val) < tolerance:
            raise ValueError("Derivative too small; Newton's method may not converge.")
        
        # Update step for Newton's method
        r_prime_next = r_prime - f_val / df_val
        
        # Check for convergence
        if abs(r_prime_next - r_prime) < tolerance:
            L1_distance = r_prime_next * R
            print(f"Approximate L1 distance from the primary body = {L1_distance:.4e} meters")
            print(f"Converged in {iteration + 1} iterations.\n")
            return L1_distance
        
        r_prime = r_prime_next

    raise ValueError("Maximum iterations exceeded; Newton's method did not converge.")

# Moon-Earth system parameters
find_L1_distance_jax(
    M=5.974e24,  # Mass of Earth in kg
    m=7.348e22,  # Mass of Moon in kg
    R=3.844e8,   # Distance between Earth and Moon in meters
    initial_guess=0.9,
    system_name="Moon-Earth"
)

# Earth-Sun system parameters
find_L1_distance_jax(
    M=1.989e30,  # Mass of Sun in kg
    m=5.974e24,  # Mass of Earth in kg
    R=1.496e11,  # Distance between Earth and Sun in meters
    initial_guess=0.99,
    system_name="Earth-Sun"
)

# Jupiter-Sun system parameters (with Jupiter at Earth's distance from the Sun)
find_L1_distance_jax(
    M=1.989e30,  # Mass of Sun in kg
    m=1.898e27,  # Mass of Jupiter in kg
    R=1.496e11,  # Distance between Jupiter and Sun in meters
    initial_guess=0.99,
    system_name="Jupiter-Sun"
)
