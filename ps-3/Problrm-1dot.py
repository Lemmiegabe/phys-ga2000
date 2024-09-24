import numpy as np
import numpy as np
import time
import matplotlib.pyplot as plt

# Function to perform matrix multiplication using np.dot and measure the time taken
def measure_time(N):
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    
    start_time = time.time()
    
    C = np.dot(A, B)
                
    end_time = time.time()
    
    return end_time - start_time

# List of matrix sizes to test from 10 to 1000
matrix_sizes = list(range(10, 1001, 10))  # Step size of 10 (e.g., 10, 20, 30, ..., 200)
computation_times = []

# Measure computation time for each matrix size
for size in matrix_sizes:
    time_taken = measure_time(size)
    computation_times.append(time_taken)
   

# Theoretical N^3 curve for comparison (normalized)
n_cubed = [(size**3)/4.1e10 for size in matrix_sizes]  # Scaled to fit on the same plot

# Plotting the results
plt.figure(figsize=(10, 10))
plt.plot(matrix_sizes, computation_times, marker='o', label='Empirical Computation Time')
plt.plot(matrix_sizes, n_cubed, linestyle='--', color='red', label=r'Theoretical $N^3$')

plt.title('Matrix Size vs Computation Time')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Computation Time (seconds)')
plt.legend()
plt.grid(False)
#plt.savefig('matrix_multiplication2.png')
plt.show()