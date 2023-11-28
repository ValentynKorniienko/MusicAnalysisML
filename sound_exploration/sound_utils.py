import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(file_path):
    """
    Load an audio file as a waveform and a sampling rate.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        tuple: waveform and sampling rate
    """
    data, sr = librosa.load(file_path)
    return data, sr


def plot_waveform(data, sr, title="Sound Waves"):
    """
    Plot the waveform of the audio data.

    Args:
        data (array): Audio data.
        sr (int): Sampling rate.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y=data, sr=sr, color="#A300F9")
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_fourier_transform(data, sr, title="Fourier Transform"):
    """
        Plot the Fourier Transform of the audio data.

        Args:
            data (array): Audio data.
            sr (int): Sampling rate.
            title (str): Title of the plot.
        """
    D = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
    plt.figure(figsize=(16, 6))
    plt.plot(D)
    plt.title(title)
    plt.show()


def plot_spectrogram(data, sr, title="Spectrogram"):
    """
    Plot the spectrogram of the audio data.

    Args:
        data (array): Audio data.
        sr (int): Sampling rate.
        title (str): Title of the plot.
    """
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_mel_spectrogram(data, sr, title="Mel Spectrogram"):
    """
    Plot the Mel spectrogram of the audio data.

    Args:
        data (array): Audio data.
        sr (int): Sampling rate.
        title (str): Title of the plot.
    """
    S = librosa.feature.melspectrogram(y=data, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def plot_mfcc(data, sr, title="MFCC"):
    """
        Plot the MFCC of the audio data.

        Args:
            data (array): Audio data.
            sr (int): Sampling rate.
            title (str): Title of the plot.
        """
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
