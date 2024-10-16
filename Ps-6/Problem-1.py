import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

# Create a directory to save the figures if it doesn't exist
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
# Adjusting figure sizes and legend font size
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['legend.fontsize'] = 'large'

# (a) Read the data
# Open the FITS file
hdu_list = fits.open('specgrid.fits')

# Extract the log wavelength and flux data
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# Convert log wavelength to regular wavelength (Angstroms)
wavelength = 10**logwave

# Plot a few spectra (e.g., first 5)
for i in range(5):
    plt.plot(wavelength, flux[i], label=f'Galaxy {i+1}')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (10^-17 erg/s/cm^2/A)')
plt.title("Sample Galaxy Spectra")
plt.legend()  # Add legend to distinguish galaxies
plt.savefig('sample_galaxy_spectra.png')
plt.show()

# (b) Normalize the fluxes
# Integrate fluxes and normalize
norm_flux = np.trapz(flux, x=wavelength, axis=1)
flux_normalized = flux / norm_flux[:, None]

# (c) Subtract the mean spectrum
# Calculate mean spectrum
mean_spectrum = np.mean(flux_normalized, axis=0)

# Subtract mean from each spectrum
residuals = flux_normalized - mean_spectrum

# (d) Perform PCA by finding covariance matrix
# Calculate covariance matrix
cov_matrix = np.cov(residuals, rowvar=False)

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Plot the first 5 eigenvectors
for i in range(5):
    plt.plot(wavelength, eigenvectors[:, i], label=f'Eigenvector {i+1}')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Eigenvector value')
plt.title('First 5 Eigenvectors')
plt.legend()  # Add legend for eigenvectors
plt.savefig('first_5_eigenvectors.png')
plt.show()

# (e) Perform PCA using SVD
# Perform SVD on residuals matrix
U, S, Vt = np.linalg.svd(residuals, full_matrices=False)


# Extract right singular vectors (V)
V = Vt.T

# Compute the covariance matrix R^T R
cov_matrix = np.dot(residuals.T, residuals)

# Step 4: Compare the eigenvectors from PCA and SVD
# Perform PCA (which also computes R^T R internally)
pca = PCA(n_components=residuals.shape[1])
pca.fit(residuals)

# The PCA eigenvectors are equivalent to the right singular vectors from SVD
eigenvectors_pca = pca.components_.T  # Transposed to match V's shape from SVD

# Check if the eigenvectors from SVD and PCA are equivalent (up to numerical precision)
comparison = np.allclose(V, eigenvectors_pca)
print(f"Are the eigenvectors from SVD and PCA equivalent? {comparison}")


# Print condition numbers for further insight
cond_pca = np.linalg.cond(pca.components_.T)
cond_svd = np.linalg.cond(V)

print(f"Condition number of PCA eigenvectors: {cond_pca}")
print(f"Condition number of SVD eigenvectors: {cond_svd}")

# Vt contains the eigenvectors of the covariance matrix
# Compare with the eigenvectors from the covariance method
for i in range(5):
    plt.plot(wavelength, Vt[i], label=f'SVD Eigenvector {i+1}')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('SVD Eigenvector value')
plt.title('First 5 SVD Eigenvectors')
plt.legend()  # Add legend for SVD eigenvectors
plt.savefig('first_5_svd_eigenvectors.png')
plt.show()

# (f) Compare condition numbers
cond_C = np.linalg.cond(cov_matrix)
cond_R = np.linalg.cond(residuals)
print("Condition number of C:", cond_C)
print("Condition number of R:", cond_R)

# (g) Compressing the spectra using Nc = 5
Nc = 5  # You may choose to vary this number
approx_spectra = np.dot(np.dot(U[:, :Nc], np.diag(S[:Nc])), Vt[:Nc, :])

# Add back the mean and multiply by normalization
approx_spectra = approx_spectra + mean_spectrum
approx_spectra = approx_spectra * norm_flux[:, None]

# (h) Plot c0 vs c1 and c0 vs c2
c0 = U[:, 0] * S[0]
c1 = U[:, 1] * S[1]
c2 = U[:, 2] * S[2]

plt.scatter(c0, c1, marker='.', label='c0 vs c1')
plt.xlabel('c0')
plt.ylabel('c1')
plt.title('c0 vs c1')
plt.legend()  # Add legend for c0 vs c1 plot
plt.savefig('c0_vs_c1.png')
plt.show()

plt.scatter(c0, c2, marker='.', label='c0 vs c2')
plt.xlabel('c0')
plt.ylabel('c2')
plt.title('c0 vs c2')
plt.legend()  # Add legend for c0 vs c2 plot
plt.savefig('c0_vs_c2.png')
plt.show()

# (i) Residuals as a function of Nc
residuals_list = []
for Nc in range(1, 21):
    approx_spectra = np.dot(np.dot(U[:, :Nc], np.diag(S[:Nc])), Vt[:Nc, :])
    approx_spectra = approx_spectra + mean_spectrum
    approx_spectra = approx_spectra * norm_flux[:, None]
    
    # Calculate residuals
    residuals = np.sqrt(np.mean((flux - approx_spectra)**2, axis=1))
    residuals_list.append(np.mean(residuals))

# Plot residuals as a function of Nc
plt.plot(range(1, 21), residuals_list, label='Mean Squared Residuals')
plt.xlabel('Nc')
plt.ylabel('Mean squared residuals')
plt.title('Residuals as a function of Nc')
plt.legend()  # Add legend for residuals plot
plt.savefig('residuals_vs_Nc.png')
plt.show()

# Set Nc to 20 and reconstruct the spectra
Nc = 30

approx_spectra = np.dot(np.dot(U[:, :Nc], np.diag(S[:Nc])), Vt[:Nc, :])

# Inverse PCA transformation to recover spectra in original space
approx_spectra_original_space = pca.inverse_transform(approx_spectra)

# Add back the mean spectrum and reapply the original normalization
approx_spectra_original_space = approx_spectra_original_space + mean_spectrum
approx_spectra_original_space = approx_spectra_original_space * norm_flux[:, None]

# Calculate the residuals (difference between original and approximated spectra)
residuals = flux - approx_spectra_original_space

# Compute the RMS residual
# Squared residuals for each spectrum, then mean, and square root
squared_residuals = np.mean((residuals) ** 2, axis=1)
rms_residual = np.sqrt(np.mean(squared_residuals))

# Print the RMS residual
print(f"Root-Mean Squared Residual for Nc = 20: {rms_residual}")

# Plot the first galaxy's original spectrum vs the approximated spectrum (using Nc = 5)
Nc = 5  # Reassign Nc to 5 before plotting the approximation
approx_spectra = np.dot(np.dot(U[:, :Nc], np.diag(S[:Nc])), Vt[:Nc, :])
approx_spectra = approx_spectra + mean_spectrum
approx_spectra = approx_spectra * norm_flux[:, None]

# First galaxy original spectrum (galaxy 1, index 0)
original_spectrum = flux[0]

# First galaxy approximated spectrum
approx_spectrum_galaxy1 = approx_spectra[0]

# Plot both the original and approximated spectrum
plt.plot(wavelength, original_spectrum, label='Original Spectrum (Galaxy 1)')
plt.plot(wavelength, approx_spectrum_galaxy1, label=f'Approximated Spectrum (Galaxy 1, Nc={Nc})', linestyle='--')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (10^-17 erg/s/cm^2/A)')
plt.title('Original vs Approximate Spectrum (Galaxy 1)')
plt.legend()  # Add legend to distinguish original vs approximated spectrum
plt.savefig('original_vs_approximated_spectrum.png')
plt.show()
