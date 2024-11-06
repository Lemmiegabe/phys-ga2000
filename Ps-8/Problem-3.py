import numpy as np
import matplotlib.pyplot as plt

# Load the data
dow_data = np.loadtxt('dow.txt')

# Define the number of data points
N = len(dow_data)

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(dow_data, label="Original Data")
plt.title("Dow Jones Industrial Average (2006 - 2010)")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.legend()
plt.savefig("output_original.png")
plt.show()

# Compute the Fourier transform using rfft
fft_dow = np.fft.rfft(dow_data)

# Retain the first 10% of Fourier coefficients
fft_dow_10 = np.copy(fft_dow)
fft_dow_10[int(0.1 * len(fft_dow)):] = 0

# Inverse Fourier transform with 10% of coefficients
dow_data_10 = np.fft.irfft(fft_dow_10, n=N)

# Plot the original and 10%-filtered data
plt.figure(figsize=(12, 6))
plt.plot(dow_data, label="Original Data")
plt.plot(dow_data_10, label="10% Fourier Coefficients")
plt.title("Original vs. 10% Low-Frequency Components")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.legend()
plt.savefig("output_fourier10.png")
plt.show()

# Retain only the first 2% of Fourier coefficients
fft_dow_2 = np.copy(fft_dow)
fft_dow_2[int(0.02 * len(fft_dow)):] = 0

# Inverse Fourier transform with 2% of coefficients
dow_data_2 = np.fft.irfft(fft_dow_2, n=N)

# Plot the original and 2%-filtered data
plt.figure(figsize=(12, 6))
plt.plot(dow_data, label="Original Data")
plt.plot(dow_data_2, label="2% Fourier Coefficients")
plt.title("Original vs. 2% Low-Frequency Components")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.legend()
plt.savefig("output_fourier2.png")
plt.show()
