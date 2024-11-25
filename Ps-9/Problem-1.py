import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Part (a): Simple Harmonic Oscillator
def harmonic_oscillator(t, y, omega):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Part (c): Anharmonic Oscillator
def anharmonic_oscillator(t, y, omega):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x**3
    return [dxdt, dvdt]

# Part (e): Van der Pol Oscillator
def van_der_pol_oscillator(t, y, mu, omega):
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return [dxdt, dvdt]

# Parameters
omega = 1
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Part (a) - Simple Harmonic Oscillator
y0_a = [1, 0]  # Initial conditions
solution_a = solve_ivp(harmonic_oscillator, t_span, y0_a, args=(omega,), t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(solution_a.t, solution_a.y[0], label="x(t)")
plt.title("Part (a): Simple Harmonic Oscillator")
plt.xlabel("Time t")
plt.ylabel("Displacement x")
plt.grid()
plt.legend()
plt.savefig("Part (a) Simple Harmonic Oscillator.png")
plt.show()

# Part (b) - Increased Amplitude
y0_b = [2, 0]  # Increased amplitude
solution_b = solve_ivp(harmonic_oscillator, t_span, y0_b, args=(omega,), t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(solution_b.t, solution_b.y[0], label="x(t) with larger amplitude")
plt.title("Part (b): Simple Harmonic Oscillator (Increased Amplitude)")
plt.xlabel("Time t")
plt.ylabel("Displacement x")
plt.grid()
plt.legend()
plt.savefig("Part (b) Simple Harmonic Oscillator (Increased Amplitude).png")
plt.show()

# Part (c) - Anharmonic Oscillator
y0_c = [1, 0]
solution_c = solve_ivp(anharmonic_oscillator, t_span, y0_c, args=(omega,), t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(solution_c.t, solution_c.y[0], label="x(t)")
plt.title("Part (c): Anharmonic Oscillator")
plt.xlabel("Time t")
plt.ylabel("Displacement x")
plt.grid()
plt.legend()
plt.savefig("Part (c) Anharmonic Oscillator.png")
plt.show()
# Part (c) - Anharmonic Oscillator
y0_c = [2, 0]
solution_c = solve_ivp(anharmonic_oscillator, t_span, y0_c, args=(omega,), t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(solution_c.t, solution_c.y[0], label="x(t)")
plt.title("Part (c): Anharmonic Oscillator(Increased Amplitude)")
plt.xlabel("Time t")
plt.ylabel("Displacement x")
plt.grid()
plt.legend()
plt.savefig("Part (c) Anharmonic Oscillator(Increased Amplitude).png")
plt.show()

# Part (d) - Phase Space for Anharmonic Oscillator
plt.figure(figsize=(8, 5))
plt.plot(solution_c.y[0], solution_c.y[1], label="Phase Space")
plt.title("Part (d): Anharmonic Oscillator Phase Space")
plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.grid()
plt.legend()
plt.savefig("Part (d) Anharmonic Oscillator Phase Space.png")
plt.show()

# Part (e) - Van der Pol Oscillator
mu_values = [1, 2, 4]  # Different values of mu
t_span_vdp = (0, 20)
t_eval_vdp = np.linspace(t_span_vdp[0], t_span_vdp[1], 1000)
y0_e = [1, 0]  # Initial conditions

plt.figure(figsize=(10, 6))
for mu in mu_values:
    solution_e = solve_ivp(van_der_pol_oscillator, t_span_vdp, y0_e, args=(mu, omega), t_eval=t_eval_vdp)
    plt.plot(solution_e.y[0], solution_e.y[1], label=f"μ = {mu}")

plt.title("Part (e): Van der Pol Oscillator Phase Space")
plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.grid()
plt.legend()
plt.savefig("Part (e) Van der Pol Oscillator Phase Space.png")
plt.show()
