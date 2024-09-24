import numpy as np

import matplotlib.pyplot as plt

# Constants
tau = 3.053 / np.log(2)  # Convert half-life to decay constant

# Generate 1000 random numbers from the given nonuniform distribution
random_numbers = np.random.rand(1000)
decay_times = -tau * np.log(1 - random_numbers)

# Sort the decay times
sorted_decay_times = np.sort(decay_times)

# Calculate the number of atoms that have not decayed as a function of time
time_points = np.linspace(0, max(sorted_decay_times), 1000)
remaining_atoms = [np.sum(sorted_decay_times > t) for t in time_points]

# Plot the results
plt.plot(time_points, remaining_atoms)
plt.xlabel('Time (min)')
plt.ylabel('Number of atoms not decayed')
plt.title('Decay of 1000 atoms of 209 Tl')
plt.grid(True)
plt.savefig('decay_simulation.png')
plt.show()