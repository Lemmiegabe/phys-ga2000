import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
hbar = 1.0545718e-34  # Planck's constant (Js)
m = 9.10938356e-31    # Mass of the electron (kg)
L = 1e-8              # Length of the box (m)
N = 1000              # Number of spatial grid points
a = L / N             # Grid spacing
x = np.linspace(0, L, N)  # Spatial grid
h = 1e-18             # Time step (s)

# Initial wavefunction parameters
x0 = L * 0.5          # Initial position of the wave packet (near x = 0)
sigma = 1e-10         # Width of the wave packet
kappa = 5e10          # Wavenumber

# Initialize the wavefunction
psi = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)
psi[0] = psi[-1] = 0  # Boundary conditions (psi = 0 at x = 0 and x = L)

# Coefficients for the tridiagonal matrices A and B
r = 1j * hbar * h / (2 * m * a**2)
a1 = 1 + 2 * r
a2 = -r
b1 = 1 - 2 * r
b2 = r

# Create the A and B matrices (tridiagonal in banded form)
A = np.zeros((3, N), dtype=complex)
B = np.zeros((3, N), dtype=complex)

A[1, :] = a1
A[0, 1:] = a2
A[2, :-1] = a2

B[1, :] = b1
B[0, 1:] = b2
B[2, :-1] = b2

# Function for one Crank-Nicolson step
def crank_nicolson_step(psi, A, B):
    # Compute v = B @ psi (matrix-vector product for B)
    v = B[1] * psi + np.roll(B[0], -1) * np.roll(psi, -1) + np.roll(B[2], 1) * np.roll(psi, 1)
    # Solve A @ psi_new = v using solve_banded
    psi_new = solve_banded((1, 1), A, v)
    return psi_new

# Simulation parameters
num_steps = 1000  # Number of time steps
wavefunction_history = [psi.copy()]  # Store wavefunction at each step

# Perform multiple Crank-Nicolson steps
for _ in range(num_steps):
    psi = crank_nicolson_step(psi, A, B)
    wavefunction_history.append(psi.copy())

# Save snapshots at selected time steps
snapshot_indices = [0, 200, 400, 600, 800, 1000]  # Time steps for snapshots
for i, idx in enumerate(snapshot_indices):
    plt.figure(figsize=(8, 5))
    plt.plot(x, np.real(wavefunction_history[idx]), label=f"Re[ψ(x, t)] at step {idx}", color="blue")
    plt.xlabel("x (m)")
    plt.ylabel("Re[ψ(x, t)]")
    plt.title(f"Wavefunction Snapshot at Step {idx}")
    plt.grid()
    plt.legend()
    plt.savefig(f"wavefunction_snapshot_step_{idx}.png")  # Save the snapshot as PNG
    plt.close()

# Animation of the wavefunction
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(wavefunction_history[0]), color="blue")
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (m)")
ax.set_ylabel("Re[ψ]")
ax.set_title("Wavefunction Evolution (Real Part)")

def update(frame):
    line.set_ydata(np.real(wavefunction_history[frame]))
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(wavefunction_history), interval=50, blit=True)
plt.show()
