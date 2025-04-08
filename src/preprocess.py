import numpy as np
import pandas as pd

#Audio utils

def load_audio(path,sr=None):
    """
    Load an audio file and return the data and sampling rate.
    
    Parameters:
    path (str): Path to the audio file.
    sr (int): Sampling rate. If None, the original sampling rate is used.

    """
    import librosa
    try:
        y,sr=librosa.load(path,sr=sr)
        return y,sr
    except librosa.util.exceptions.ParameterError as e:
        print(f"Librosa error processing file {path}: {e}")
        return None, None
    except FileNotFoundError as e:
        print(f"File not found: {path}. Error: {e}")
        return None, None


def plot_waveform(y,sr):
    """
    Plot the waveform of an audio signal 
    
    Parameters:
    y (ndarray): Audio signal.
    sr (int): Sampling rate.

    """
    import librosa.display
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y,sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_spectogram(y,sr):
    """
    Plot the spectogram of an audio signal
    Parameters:
    y (ndarray): Audio signal.
    sr (int): Sampling rate.

    """
    import librosa.display
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 5))
    D=librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(1, 2, 1)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

#Data preprocessing utils

def standardize(X):
    """
    Standardize the features by removing the mean and scaling to unit variance.
    Parameters:
    X (ndarray): Feature matrix.
    Returns:
    ndarray: Standardized feature matrix.
    """
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled


def reduce_pca(X,n_components):
    """
    Reduce the dimensionality of the feature matrix usiong PCA.
    Parameters:
    X (ndarray): Feature matrix.
    n_components (int): Number of components to keep.
    Returns:
    ndarray: Reduced feature matrix.
    """
    from sklearn.decomposition import PCA

    pca=PCA(n_components=n_components)
    X_pca=pca.fit_transform(X)
    return X_pca

__all__ = [
    "load_audio",
    "plot_waveform",
    "plot_spectogram",
    "standardize",
    "reduce_pca"
]

if __name__ == "__main__":
    import os
    test_path = "../data/raw/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
    
    if os.path.exists(test_path):
        y, sr = load_audio(test_path)
        if y is not None:
            plot_waveform(y, sr)
            plot_spectogram(y, sr)
