import librosa
import os
import pandas as pd
import numpy as np
from scipy.signal import spectrogram

fp = "..\Data\genres_original"                                                            

def mean_var_calc(input_value, output_mean, output_var):                            
    output_mean.append(np.mean(input_value))
    output_var.append(np.var(input_value))           

def feature_extractor(fp):
    song_names = []                                                             
    tempo = []
    rms_mean = []
    rms_var = []
    zero_crossing_mean = []
    zero_crossing_var = []
    centroid_mean = []
    centroid_var = []
    rolloff_mean = []
    rolloff_var = []
    bandwidth_mean = []
    bandwidth_var = []
    harmonic_mean = []
    harmonic_var = []
    per_wgt_mean = []
    per_wgt_var = []
    stft_mean = []
    stft_var = []
    mfcc = []
    label = []

    mfcc_means = {}
    mfcc_vars = {}
    for i in range(1, 21):
        mfcc_means["mfcc_mean_" + str(i)] = [i - 1]
        mfcc_vars["mfcc_var_" + str(i)] = [i - 1] 


    for foldername in os.listdir(fp):
        print(foldername)
        folder_path = os.path.join(fp, foldername)
        for i, filename in enumerate(os.listdir(folder_path)):
            print(filename)
            song_names.append(filename)

            song, sr = librosa.load(os.path.join(folder_path, filename), sr = None)

            tempo.append(float(librosa.feature.tempo(y = song, sr = sr)[0]))

            rms = librosa.feature.rms(y = song)
            mean_var_calc(rms, rms_mean, rms_var)

            zero_crossing = librosa.feature.zero_crossing_rate(y = song)
            mean_var_calc(zero_crossing, zero_crossing_mean, zero_crossing_var)

            centroid = librosa.feature.spectral_centroid(y = song, sr = sr)
            mean_var_calc(centroid, centroid_mean, centroid_var)

            rolloff = librosa.feature.spectral_rolloff(y = song, sr = sr)
            mean_var_calc(rolloff, rolloff_mean, rolloff_var)

            bandwidth = librosa.feature.spectral_bandwidth(y = song, sr = sr)
            mean_var_calc(bandwidth, bandwidth_mean, bandwidth_var)

            harmonic = librosa.effects.harmonic(y = song)
            mean_var_calc(harmonic, harmonic_mean, harmonic_var)

            freq, time, spectr = spectrogram(song)
            per_wgt = librosa.perceptual_weighting(spectr, freq)
            mean_var_calc(per_wgt, per_wgt_mean, per_wgt_var)

            stft = librosa.feature.chroma_stft(y = song, sr = sr)                     
            mean_var_calc(stft, stft_mean, stft_var)

            mfcc.append(librosa.feature.mfcc(y = song, sr = sr))
            for value in mfcc_means.values():
                value.append(np.mean(mfcc[i][value[0]]))
            for value in mfcc_vars.values():
                value.append(np.var(mfcc[i][value[0]]))

            label.append(filename[0:-10])

    for value in mfcc_means.values():
        value.pop(0)
    for value in mfcc_vars.values():
        value.pop(0)

    features = {"tempo": tempo, "rms_mean": rms_mean, "rms_var" : rms_var, "zero_crossing_mean" : zero_crossing_mean,       #i might swap this over to a numpy array if i have the time cause it would probably be cleaner
                "zero_crossing_var" : zero_crossing_var, "centroid_mean" : centroid_mean,"centroid_var" : centroid_var, 
                "rolloff_mean" : rolloff_mean, "bandwidth_mean" : bandwidth_mean, "bandwidth_var" : bandwidth_var, 
                "rolloff_var" : rolloff_var, "harmonic_mean" :harmonic_mean, "harmonic_var" : harmonic_var, "per_wgt_mean" : per_wgt_mean,
                "per_wgt_var" : per_wgt_var, "stft_mean" : stft_mean, "stft_var" : stft_var, "label" : label}
                
    features.update(mfcc_means)
    features.update(mfcc_vars)

    feature_frame = pd.DataFrame(data = features, index = song_names)
    feature_frame.to_excel("../features_30sec.xlsx")

feature_extractor(fp)                                                   