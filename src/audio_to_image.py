import librosa
import numpy as np 
import matplotlib.pyplot as plt

def audio_to_mel_array(file_path,sr=22050,n_mels=128,hop_length=512):
    try:

        y,sr=librosa.load(file_path,sr=sr)
        S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=n_mels,hop_length=hop_length)
        S_dB=librosa.power_to_db(S,ref=np.max)

        #Normalize the mel spectrogram
        S_norm=((S_dB-S_dB.min())/(S_dB.max()-S_dB.min()))

        return np.expand_dims(S_norm,axis=-1)
    except Exception as e:
        print(f"[SKIP] Fichier corrompu ou incompatible : {file_path} → {e}")
        return None
    
def show_mel_image(img_array):
    plt.figure(figsize=(10, 4))
    plt.imshow(img_array.squeeze(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (mel)')
    plt.show()

def pad_spectrogram(img,target_width=143):
    current_width=img.shape[1]
    if current_width>= target_width:
        return img[:,:target_width,:]
    else:
        pad_width=target_width-current_width
        left_pad=pad_width//2
        right_pad=pad_width-left_pad
        padded_img=np.pad(img,((0,0),(left_pad,right_pad),(0,0)),mode='constant',constant_values=0)
        return padded_img
