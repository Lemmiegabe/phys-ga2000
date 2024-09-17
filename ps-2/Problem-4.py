import numpy as np
import matplotlib.pyplot as plt

 
# setting parameters (these values can be changed)
xDomain, yDomain = np.linspace(-2,2,500,dtype=np.float32), np.linspace(-2,2,500,dtype=np.float32)
bound = 2
power = 2             # any positive floating point value (n)
max_iterations = 50   # any positive integer value
colormap = 'magma'    # set to any matplotlib valid colormap
 
 
 # computing 2-d array to represent the mandelbrot-set
iterationArray = []
for y in yDomain:
    row = []
    for x in xDomain:
        c = complex(x,y)
        z = 0
        for i in range(max_iterations):
            if(abs(z) >= bound):
                row.append(i)
                break
            else: z = z**power + c
        else:
            row.append(0)
 
    iterationArray.append(row)
 
# plotting the data
ax = plt.axes()
ax.set_aspect('equal')
graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
plt.colorbar(graph)
plt.xlabel("Real-Axis")
plt.ylabel("Imaginary-Axis")
plt.title('Mandelbrot set')
plt.gcf().set_size_inches(10,10)

plt.savefig('Mandelbrot.png',format='png')
plt.show()