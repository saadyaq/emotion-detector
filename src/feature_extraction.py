import pandas as pd
import numpy as np
import librosa

def extract_features(file_path):
    """
    Extracts features from an audio file.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        dict: A dictionnary containing exrtracted features"""
    
    features={}

    try:

        #load the file

        y,sr=librosa.load(file_path,sr=None)
        #extract features

        features['chroma_stft']=np.mean(librosa.feature.chroma_stft(y=y,sr=sr))
        features['chroma_cqt']=np.mean(librosa.feature.chroma_cqt(y=y,sr=sr))
        features['mfcc']=np.mean(librosa.feature.mfcc(y=y,sr=sr))
        features['spectral_centroid']=np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
        features['spectral_bandwidth']=np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr))
        features['spectral_contrast']=np.mean(librosa.feature.spectral_contrast(y=y,sr=sr)) 
        features['rms']=np.mean(librosa.feature.rms(y=y)) 
        features['zero_crossing_rate']=np.mean(librosa.feature.zero_crossing_rate(y=y))
        features['tempo']=np.mean(librosa.beat.tempo(y=y,sr=sr))
        return list(features.values())   
    except librosa.util.exceptions.ParameterError as e:
        print(f"Librosa error processing file {file_path}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {file_path}. Error: {e}")
        return None
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    


