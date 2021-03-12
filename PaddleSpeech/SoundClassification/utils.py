import numpy as np
import librosa
def melspect(y,sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                window = 'hann',
                center = True,
                pad_mode = 'reflect',
                ref = 1.0,
                amin = 1e-10,
                top_db = None):
    
    s = librosa.stft(y,n_fft=window_size, 
                               hop_length=hop_size,
                               win_length=window_size, 
                               window=window, 
                               center=center, pad_mode=pad_mode)

    power = np.abs(s)**2
    melW = librosa.filters.mel(sr=sample_rate, 
                               n_fft=window_size,
                               n_mels=mel_bins,
                fmin=fmin, fmax=fmax)
    mel = np.matmul(melW,power)
    db = librosa.power_to_db(mel,ref=ref,amin=amin,top_db=None)
    db = db.transpose()
    return db


    
    
    
