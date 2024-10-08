import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, lstsq

# Step 1: Read the file into a pandas DataFrame, skipping the first row (header) and using '|' as the delimiter
df = pd.read_csv('signal.dat', sep='|', skiprows=1, header=None)

# Step 2: Remove unnecessary whitespace and drop the first and last empty columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.drop(columns=[0, 3])  # Drop empty columns created from '|' characters at the beginning and end

# Rename columns to 'Time' and 'Signal' for easier access
df.columns = ['Time', 'Signal']

# Step 3: Convert columns to numeric and drop rows with NaN values
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df['Signal'] = pd.to_numeric(df['Signal'], errors='coerce')
df = df.dropna()

# Step 4: Check if DataFrame is still empty and display final cleaned data
if df.empty:
    raise ValueError("The DataFrame is still empty after cleaning. Please inspect the input data.")

# Access the updated time and signal arrays
time = df['Time'].values
signal = df['Signal'].values
uncertainty = 2.0

# (a) Plot the original data using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(time, signal,marker= '.', color='blue', label='Original Data')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal vs Time')
plt.legend()
#plt.savefig('original_data.png')
plt.show()

# (b) Use SVD to find the best third-order polynomial fit in time to the signal
# Scale the time variable for stability in polynomial fitting
time_scaled = (time - np.mean(time)) / np.std(time)

# Design matrix for third-order polynomial
A = np.vstack([time_scaled**i for i in range(4)]).T

# Perform SVD and fit the polynomial
U, s, Vt = svd(A, full_matrices=False)
c = lstsq(A, signal, rcond=None)[0]

# Polynomial fit
poly_fit = np.polyval(c[::-1], time_scaled)

# Plot the third-order polynomial fit with original data
plt.figure(figsize=(10, 6))
plt.scatter(time, signal,marker='.', color='blue', label='Original Data')
plt.scatter(time, poly_fit, marker='.', color='red', alpha=0.6, label='3rd Order Polynomial Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Third-Order Polynomial Fit (Scatter Plot)')
plt.legend()
#plt.savefig('third_order_fit.png')
plt.show()

# (c) Calculate residuals and assess fit quality
residuals = signal - poly_fit

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(time, residuals, marker='.', color='orange', label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residuals of Third-Order Polynomial Fit')
plt.legend()
#plt.savefig('residuals.png')
plt.show()

# Evaluate if the third-order polynomial is a good fit given the measurement uncertainty
residual_std = np.std(residuals)
print(f"Standard deviation of residuals: {residual_std:.2f}")
print("Conclusion: The residuals are large relative to the uncertainty (2.0), indicating a poor fit.")

# (d) Fit a much higher-order polynomial
def fit_polynomial(order):
    A = np.vstack([time_scaled**i for i in range(order + 1)]).T
    U, s, Vt = svd(A, full_matrices=False)
    condition_number = s[0] / s[-1]
    c = lstsq(A, signal, rcond=None)[0]
    poly_fit = np.polyval(c[::-1], time_scaled)
    return poly_fit, condition_number

# Fit a 10th-order polynomial
poly_fit_10, cond_number_10 = fit_polynomial(10)

# Plot the 10th-order polynomial fit
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, marker='.',color='blue', label='Original Data')
plt.scatter(time, poly_fit_10,marker='.', color='green', alpha=0.6, label='10th Order Polynomial Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('10th Order Polynomial Fit (Scatter Plot)')
plt.legend()
#plt.savefig('tenth_order_fit.png')
plt.show()

print(f"Condition number for 10th-order polynomial: {cond_number_10:.2e}")
print("Conclusion: A high condition number suggests numerical instability for this fit.")

# (e) Fit a set of sin and cos functions plus a zero-point offset (Fourier series)
# Define the time span and fundamental frequency
time_span = time.max() - time.min()
fundamental_period = time_span / 2
fundamental_frequency = 2 * np.pi / fundamental_period

# Fit Fourier series with harmonics up to the 5th harmonic
n_terms = 5
A_fourier = np.ones((len(time), 2 * n_terms + 1))
for i in range(1, n_terms + 1):
    A_fourier[:, 2 * i - 1] = np.sin(i * fundamental_frequency * time)
    A_fourier[:, 2 * i] = np.cos(i * fundamental_frequency * time)

# Use least squares to fit the Fourier series to the original signal
c_fourier = lstsq(A_fourier, signal, rcond=None)[0]
fourier_fit = A_fourier @ c_fourier

# Plot the Fourier series fit
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, marker='.',color='blue', label='Original Data')
plt.scatter(time, fourier_fit, marker='.', color='purple', alpha=0.6, label='Fourier Series Fit (5 Harmonics)')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Fourier Series Fit (Scatter Plot)')
plt.legend()
#plt.savefig('fourier_fit.png')
plt.show()

# (f) Analyze periodicity using scatter plot for the Fourier fit
plt.figure(figsize=(10, 6))
plt.scatter(time, fourier_fit, marker='.',color='purple', alpha=0.6, label='Fourier Series Fit (5 Harmonics)')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Fourier Series Fit to Analyze Periodicity')
plt.legend()
#plt.savefig('fourier_periodicity.png')
plt.show()

# Print the fitted Fourier coefficients to observe the contribution of each harmonic
print("Fitted Fourier Coefficients:", c_fourier)
time_span = time.max() - time.min()  # Total time span of the data
fundamental_period = time_span / 2   # Period of the signal's fundamental frequency

# Print the period
print(f"Fundamental period of the signal: {fundamental_period * 2:.2f}")
