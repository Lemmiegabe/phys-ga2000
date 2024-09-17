import numpy as np
import timeit

def calculate_M():
    L = 100.0
    ran = np.arange(-L, L+1)
    i,j,k = np.meshgrid(ran, ran, ran)
    M = np.where((i != 0) | (j != 0) | (k != 0),
                 (-1.0)**(i + j + k + 1 )/np.sqrt(i ** 2 + j ** 2 + k ** 2),
                 0).sum()
    return M

# Measure the execution time
execution_time = timeit.timeit(calculate_M, number=1)
print(f"Execution time: {execution_time} seconds")

#print(M)