# pip install numpy scipy matplotlib librosa streamlit soundfile
# streamlit run steganosaurus_beeg.py

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
import soundfile as sf
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import get_window

# Function to load audio file
def load_audio(filename):
    y, sr = librosa.load(filename, sr=None, mono=True)  # Load as mono for simplicity
    return y, sr

# Function to compute Goertzel algorithm
def goertzel_algorithm(y, sr, target_freqs, N, overlap):
    results = {}
    step = N - overlap
    num_windows = int((len(y) - N) / step) + 1
    window = get_window('hann', N)

    for freq in target_freqs:
        k = int(0.5 + (N * freq) / sr)
        omega = (2 * np.pi * k) / N
        cosine = np.cos(omega)
        sine = np.sin(omega)
        coeff = 2 * cosine
        s_prev = 0
        s_prev2 = 0
        power = []

        for i in range(num_windows):
            start = i * step
            end = start + N
            if end > len(y):
                break
            data = y[start:end] * window
            for sample in data:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            result = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
            power.append(result)
            s_prev = 0
            s_prev2 = 0

        results[freq] = power

    return results

# Function for linear prediction error
def linear_prediction_error(y, order=4):
    # Autocorrelation method
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    R = autocorr[:order+1]
    # Solve Yule-Walker Equations
    from scipy.linalg import toeplitz, solve_toeplitz
    r = R[1:]
    R_matrix = toeplitz(R[:-1])
    a = np.linalg.inv(R_matrix).dot(r)
    error = R[0] - np.dot(a, r)
    return error, a

# Function to compute statistical metrics
def compute_statistics(y):
    stats = {}
    stats['Mean'] = np.mean(y)
    stats['Variance'] = np.var(y)
    stats['Skewness'] = skew(y)
    stats['Kurtosis'] = kurtosis(y)
    stats['Energy'] = np.sum(y ** 2)
    return stats

# Function to compute AQMs
def compute_aqm(y_original, y_modified):
    # Mean Squared Error (MSE)
    mse = np.mean((y_original - y_modified) ** 2)
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(y_original ** 2)
    noise_power = np.mean((y_original - y_modified) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return {'MSE': mse, 'SNR': snr}

# Function to plot waveform
def plot_waveform(y, sr, title='Waveform'):
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

# Function to plot frequency spectrum
def plot_frequency_spectrum(y, sr, title='Frequency Spectrum'):
    N = len(y)
    Y = np.abs(fft(y))[:N//2]
    freqs = fftfreq(N, 1/sr)[:N//2]
    plt.figure(figsize=(14, 4))
    plt.plot(freqs, Y)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    st.pyplot(plt.gcf())
    plt.clf()

# Function to plot spectrogram
def plot_spectrogram(y, sr, title='Spectrogram'):
    plt.figure(figsize=(14, 4))
    S = np.abs(librosa.stft(y))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

# Function to plot Goertzel results
def plot_goertzel_results(results, sr, N, overlap):
    step = N - overlap
    time_axis = np.arange(len(next(iter(results.values())))) * (step / sr)
    plt.figure(figsize=(14, 4))
    for freq, power in results.items():
        plt.plot(time_axis, power, label=f'{freq} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.legend()
    plt.title('Goertzel Algorithm Results')
    st.pyplot(plt.gcf())
    plt.clf()

# Main function with Streamlit
def main():
    st.title('Audio Steganalysis Tool')

    st.sidebar.header('Audio File Selection')
    original_file = st.sidebar.file_uploader('Upload Original Audio File', type=['wav', 'mp3'])
    modified_file = st.sidebar.file_uploader('Upload Modified Audio File', type=['wav', 'mp3'])

    if original_file and modified_file:
        y_orig, sr_orig = load_audio(original_file)
        y_mod, sr_mod = load_audio(modified_file)

        # Ensure sampling rates are the same
        if sr_orig != sr_mod:
            st.error('Sampling rates of the two files do not match.')
            return

        st.sidebar.header('Analysis Parameters')
        target_freqs = st.sidebar.text_input('Target Frequencies (Hz, comma-separated)', '1000,2000,3000')
        target_freqs = [float(freq.strip()) for freq in target_freqs.split(',')]

        N = st.sidebar.number_input('Window Size (N)', min_value=100, max_value=4096, value=1024)
        overlap = st.sidebar.number_input('Overlap', min_value=0, max_value=N-1, value=N//2)

        # Analysis on Original Audio
        st.header('Original Audio Analysis')
        st.subheader('Waveform')
        plot_waveform(y_orig, sr_orig, 'Original Waveform')

        st.subheader('Frequency Spectrum')
        plot_frequency_spectrum(y_orig, sr_orig, 'Original Frequency Spectrum')

        st.subheader('Spectrogram')
        plot_spectrogram(y_orig, sr_orig, 'Original Spectrogram')

        st.subheader('Statistical Analysis')
        stats_orig = compute_statistics(y_orig)
        st.write(stats_orig)

        st.subheader('Goertzel Algorithm')
        goertzel_orig = goertzel_algorithm(y_orig, sr_orig, target_freqs, N, overlap)
        plot_goertzel_results(goertzel_orig, sr_orig, N, overlap)

        st.subheader('Linear Prediction Error')
        error_orig, coeffs_orig = linear_prediction_error(y_orig)
        st.write(f'Prediction Error: {error_orig}')
        st.write(f'Coefficients: {coeffs_orig}')

        # Analysis on Modified Audio
        st.header('Modified Audio Analysis')
        st.subheader('Waveform')
        plot_waveform(y_mod, sr_mod, 'Modified Waveform')

        st.subheader('Frequency Spectrum')
        plot_frequency_spectrum(y_mod, sr_mod, 'Modified Frequency Spectrum')

        st.subheader('Spectrogram')
        plot_spectrogram(y_mod, sr_mod, 'Modified Spectrogram')

        st.subheader('Statistical Analysis')
        stats_mod = compute_statistics(y_mod)
        st.write(stats_mod)

        st.subheader('Goertzel Algorithm')
        goertzel_mod = goertzel_algorithm(y_mod, sr_mod, target_freqs, N, overlap)
        plot_goertzel_results(goertzel_mod, sr_mod, N, overlap)

        st.subheader('Linear Prediction Error')
        error_mod, coeffs_mod = linear_prediction_error(y_mod)
        st.write(f'Prediction Error: {error_mod}')
        st.write(f'Coefficients: {coeffs_mod}')

        # Comparison and Results
        st.header('Comparison and Results')

        st.subheader('Audio Quality Metrics (AQMs)')
        aqm = compute_aqm(y_orig[:len(y_mod)], y_mod)
        st.write(aqm)

        st.subheader('Statistical Differences')
        stats_diff = {key: stats_mod[key] - stats_orig[key] for key in stats_orig}
        st.write(stats_diff)

        st.subheader('Linear Prediction Error Difference')
        error_diff = error_mod - error_orig
        st.write(f'Error Difference: {error_diff}')

        st.subheader('Goertzel Algorithm Difference')
        # Compute difference in Goertzel power
        goertzel_diff = {freq: np.array(goertzel_mod[freq]) - np.array(goertzel_orig[freq]) for freq in target_freqs}
        plot_goertzel_results(goertzel_diff, sr_orig, N, overlap)

        st.write('Areas with significant differences may indicate the presence of a watermark or hidden data.')

    else:
        st.info('Please upload both original and modified audio files to proceed.')

if __name__ == '__main__':
    main()
