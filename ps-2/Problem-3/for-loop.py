import numpy as np
import timeit

# Parameters
L = 100 # Number of atoms in all directions
a = 1    # Spacing between atoms (m)
c = 1.112e-10  # Represents 4*pi*epsilon-naught (C^2 J^-1 m^-1)
e = 1.6022e-19  # Electron charge (C)

# Initialize Madelung constant
M = 0

def calculate_madelung_constant():
    global M
    M = 0
    # Loop over lattice points
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                
                # Skip the origin
                if i == 0 and j == 0 and k == 0:
                    continue

                # Calculate the contribution of each ion
                M += (-1) ** (i + j + k) / np.sqrt(i ** 2 + j ** 2 + k ** 2)

# Measure the execution time
execution_time = timeit.timeit(calculate_madelung_constant, number=1)
print(f"Execution time: {execution_time} seconds")
#print(f"The Madelung constant of NaCl is {abs(round(M, 4))}")
