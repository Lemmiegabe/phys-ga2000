import numpy as np
import matplotlib.pyplot as plt

# Load the waveform data
piano_data = np.loadtxt('piano.txt')
trumpet_data = np.loadtxt('trumpet.txt')

# Define the sampling rate
sampling_rate = 44100  # in Hz

# Plot the waveforms
time_piano = np.arange(len(piano_data)) / sampling_rate
time_trumpet = np.arange(len(trumpet_data)) / sampling_rate

plt.figure(figsize=(14, 6))

# Piano waveform
plt.subplot(2, 1, 1)
plt.plot(time_piano, piano_data)
plt.title('Piano Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Trumpet waveform
plt.subplot(2, 1, 2)
plt.plot(time_trumpet, trumpet_data)
plt.title('Trumpet Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("output_waveform.png")
plt.show()

# Compute the Fourier transforms
fft_piano = np.fft.fft(piano_data)
fft_trumpet = np.fft.fft(trumpet_data)

# Calculate frequencies for the x-axis of the Fourier plot
frequencies = np.fft.fftfreq(len(piano_data), d=1/sampling_rate)

# Take the magnitudes of the Fourier coefficients
magnitude_piano = np.abs(fft_piano)
magnitude_trumpet = np.abs(fft_trumpet)

# Plot the magnitudes of the first 10,000 Fourier coefficients
plt.figure(figsize=(14, 6))

# Piano Fourier coefficients
plt.subplot(2, 1, 1)
plt.plot(frequencies[:10000], magnitude_piano[:10000])
plt.title('Piano Fourier Coefficients (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Trumpet Fourier coefficients
plt.subplot(2, 1, 2)
plt.plot(frequencies[:10000], magnitude_trumpet[:10000])
plt.title('Trumpet Fourier Coefficients (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.savefig("output_fourier.png")
plt.show()

# Find the fundamental frequency by locating the peak in the Fourier coefficients
fundamental_frequency_piano = frequencies[np.argmax(magnitude_piano[:10000])]
fundamental_frequency_trumpet = frequencies[np.argmax(magnitude_trumpet[:10000])]

print(f"Fundamental frequency of the piano note: {fundamental_frequency_piano:.2f} Hz")
print(f"Fundamental frequency of the trumpet note: {fundamental_frequency_trumpet:.2f} Hz")

# Determine if the notes correspond to middle C (261 Hz)
if abs(fundamental_frequency_piano - 261) < 5:
    print("The piano note is approximately middle C.")
else:
    print("The piano note is not middle C.")

if abs(fundamental_frequency_trumpet - 261) < 5:
    print("The trumpet note is approximately middle C.")
else:
    print("The trumpet note is not middle C.")
