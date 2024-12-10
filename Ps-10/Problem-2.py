import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from dcst import dst  # Replace `your_module` with the actual module name where the `dst` function is defined


# Constants
hbar = 1.0545718e-34  # Planck's constant (Js)
m = 9.10938356e-31    # Mass of the electron (kg)
L = 1e-8              # Length of the box (m)
N = 1000              # Number of grid points
x = np.linspace(0, L, N, endpoint=False)[1:]  # Exclude boundaries (0, L)

# Initial wavefunction parameters
x0 = L / 2            # Initial position of the wave packet
sigma = 1e-10         # Width of the wave packet
kappa = 5e10          # Wavenumber

# Initial wavefunction (excluding boundaries)
psi_initial = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)

# Normalize the wavefunction
a = L / N  # Grid spacing
norm_factor = np.sqrt(np.sum(np.abs(psi_initial) ** 2) * a)
psi_initial /= norm_factor

# Separate into real and imaginary parts
real_part = np.real(psi_initial)
imag_part = np.imag(psi_initial)

# Compute DST for real and imaginary parts using the custom module
alpha_k = dst(real_part) / N  # Real part coefficients
eta_k = dst(imag_part) / N    # Imaginary part coefficients

# Combine into complex coefficients b_k
b_k = alpha_k + 1j * eta_k

# Precompute energy values for each mode
k_values = np.arange(1, N)
E_k = (np.pi**2 * hbar**2 * k_values**2) / (2 * m * L**2)
omega_k = E_k / hbar  # Angular frequencies

# Function to compute the real part of the wavefunction at time t
def psi_real(x, t, alpha_k, eta_k, omega_k, k_values):
    # Reconstruct psi(x, t) using the coefficients and sine transform
    phases = np.outer(k_values, x / L * np.pi)
    cos_terms = alpha_k * np.cos(omega_k * t)
    sin_terms = eta_k * np.sin(omega_k * t)
    wave = np.sum((cos_terms + sin_terms)[:, None] * np.sin(phases), axis=0)
    
    # Normalize the wavefunction to have a maximum amplitude of 1.0
    wave /= np.max(np.abs(wave))
    return wave

# Time evolution parameters
num_steps = 500  # Number of time steps
time_steps = np.linspace(0, 1e-15, num_steps)  # Use a smaller total time range

# Precompute wavefunction values for snapshots
wavefunction_snapshots = [psi_real(x, t, alpha_k, eta_k, omega_k, k_values) for t in time_steps]

# Save snapshots at selected time steps
snapshot_times = [0, 100, 200, 300, 400]  # Indices of time steps for snapshots
for i, idx in enumerate(snapshot_times):
    plt.figure(figsize=(8, 5))
    plt.plot(x, wavefunction_snapshots[idx], color="blue")
    plt.xlabel("x (m)")
    plt.ylabel("Re[ψ(x, t)]")
    plt.title(f"Wavefunction at t = {time_steps[idx]:.1e} s")
    plt.grid()
    plt.savefig(f"wavefunction_snapshot_{i}.png")  # Save the snapshot as PNG
    plt.close()

# Compute the wavefunction at t = 10^-16 s

t_test = 1e-16
phases = np.outer(k_values, x / L * np.pi)
cos_terms = alpha_k * np.cos(omega_k * t_test)
sin_terms = eta_k * np.sin(omega_k * t_test)
psi_t = np.sum((cos_terms + sin_terms)[:, None] * np.sin(phases), axis=0)
#Normalize the the test wavefunction to have a maximum amplitude of 1.0
psi_t /= np.max(np.abs(psi_t))

# Plot the wavefunction
plt.figure(figsize=(8, 5))
plt.plot(x, psi_t, label=f"Re[ψ(x, t)] at t = {t_test:.1e} s", color="blue")
plt.xlabel("x (m)")
plt.ylabel("Re[ψ(x, t)]")
plt.title(f"Wavefunction at t = {t_test:.1e} s")
plt.grid()
plt.legend()
plt.savefig("wavefunction_t_1e-16.png")  # Save the snapshot as PNG
plt.show()

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, wavefunction_snapshots[0], color="blue")
ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)  # Fixed range since amplitude is normalized to 1.0

def update(frame):
    wave = wavefunction_snapshots[frame]
    line.set_ydata(wave)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(wavefunction_snapshots), interval=50, blit=True)

# Show animation
plt.show()
