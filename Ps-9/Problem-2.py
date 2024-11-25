import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
rho_air = 1.22  # Air density (kg/m^3)
C = 0.47  # Drag coefficient for a sphere
R = 0.08  # Radius of the cannonball (m)
m = 1.0  # Mass of the cannonball (kg)

# Cross-sectional area of the sphere
A = np.pi * R**2

# Drag force coefficient
drag_coefficient = 0.5 * rho_air * C * A / m

# Define the system of first-order ODEs
def cannonball_motion(t, y):
    x, vx, y_pos, vy = y
    v = np.sqrt(vx**2 + vy**2)  # Magnitude of velocity
    dxdt = vx
    dvxdt = -drag_coefficient * v * vx
    dydt = vy
    dvydt = -g - drag_coefficient * v * vy
    return [dxdt, dvxdt, dydt, dvydt]

# Initial conditions
v0 = 100  # Initial velocity (m/s)
theta = 30 * np.pi / 180  # Launch angle in radians
vx0 = v0 * np.cos(theta)  # Initial velocity in x-direction
vy0 = v0 * np.sin(theta)  # Initial velocity in y-direction

# Initial state: [x, vx, y, vy]
y0 = [0, vx0, 0, vy0]

# Time range
t_span = (0, 20)  # Simulate for up to 20 seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Part (a) and (b): Single trajectory with m = 1 kg
solution = solve_ivp(cannonball_motion, t_span, y0, t_eval=t_eval, events=lambda t, y: y[2])

# Extract results
x = solution.y[0]
y = solution.y[2]

# Plot trajectory for part (b)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="With Air Resistance (m = 1 kg)", color="blue")
plt.title("Cannonball Trajectory (m = 1 kg)")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.grid()
plt.legend()
plt.savefig("Cannonball Trajectory (m = 1 kg).png")
plt.show()

# Part (c): Trajectories for different masses
masses = [0.5, 1.0, 2.0]  # Different masses in kg
colors = ['red', 'blue', 'green']  # Colors for the trajectories

plt.figure(figsize=(10, 6))
for mass, color in zip(masses, colors):
    # Update the drag coefficient for each mass
    drag_coefficient = 0.5 * rho_air * C * A / mass
    
    # Solve the system for the current mass
    solution = solve_ivp(cannonball_motion, t_span, y0, t_eval=t_eval, events=lambda t, y: y[2])
    
    # Extract x and y positions
    x = solution.y[0]
    y = solution.y[2]
    
    # Plot the trajectory
    plt.plot(x, y, label=f"Mass = {mass} kg", color=color)

# Add labels, title, and legend
plt.title("Cannonball Trajectories for Different Masses")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.grid()
plt.legend()
plt.savefig("Cannonball Trajectories for Different Masses.png")
plt.show()
