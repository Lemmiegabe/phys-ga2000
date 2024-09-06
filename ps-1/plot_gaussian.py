import numpy as np
import matplotlib.pyplot as plt


#make data 
sigsq = 9
mu = 0
x = np.linspace(-10, 10, 100)
y = (1/np.sqrt(2*np.pi*sigsq))*np.exp(-0.5*(x-mu)**2/sigsq)
label = np.linspace(-10, 10, 5)
#plot data
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y,)
ax.set_title('Gaussian')
ax.set_xticks(label)
ax.text(-9, 0.11, r'$\mu=0, \sigma=3$', fontsize=12)
ax.grid(False)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
#save plot
fig.savefig('gaussian.png')
#show plot
plt.show()


