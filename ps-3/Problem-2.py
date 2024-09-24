import numpy as np
import matplotlib.pyplot as plt

# Constants
initial_bi213 = 10000
delta_t = 1  # second

# Decay constants (per second)
decay_constant_bi213_to_pb209 = 1 - 2**(-delta_t / (46 * 60))  # Bi-213 to Pb-209
decay_constant_pb209_to_bi209 = 1 - 2**(-delta_t / (3.3 * 60))  # Pb-209 to Bi-209
decay_constant_tl209_to_pb209 = 1 - 2**(-delta_t / (2.2 * 60))  # Tl-209 to Pb-209

# Initial counts
bi213_count = initial_bi213
pb209_count = 0
bi209_count = 0
tl209_count = 0

# Simulation time
total_time = 50000  # seconds

# Lists to store counts over time
time_points = []
bi213_counts = []
pb209_counts = []
bi209_counts = []
tl209_counts = []

# Simulation loop
for t in range(total_time):
    # Decay of Bi-213 to Pb-209
    decayed_bi213 = np.random.binomial(bi213_count, decay_constant_bi213_to_pb209)
    bi213_count -= decayed_bi213
    pb209_count += decayed_bi213

    # Decay of Pb-209 to Bi-209
    decayed_pb209 = np.random.binomial(pb209_count, decay_constant_pb209_to_bi209)
    pb209_count -= decayed_pb209
    bi209_count += decayed_pb209

    # Decay of Tl-209 to Pb-209
    decayed_tl209 = np.random.binomial(tl209_count, decay_constant_tl209_to_pb209)
    tl209_count -= decayed_tl209
    pb209_count += decayed_tl209

    # Store counts
    time_points.append(t)
    bi213_counts.append(bi213_count)
    pb209_counts.append(pb209_count)
    bi209_counts.append(bi209_count)
    tl209_counts.append(tl209_count)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time_points, bi213_counts, label='Bi-213')
plt.plot(time_points, pb209_counts, label='Pb-209')
plt.plot(time_points, bi209_counts, label='Bi-209')
plt.plot(time_points, tl209_counts, label='Tl-209')
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms')
plt.legend()
plt.title('Decay Simulation of Bi-213')
plt.savefig('decay_simulation.png')
plt.show()