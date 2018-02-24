"""
Module for calculating and visualizing features
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import skew, kurtosis
from sklearn.ensemble import GradientBoostingClassifier
import wfdb
from wfdb import processing

from .records import data_dir


def calc_moments(data):
    """
    Calculate moments of a 1d feature: mean, std, skew, kurtosis
    """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_skew = skew(data)
    data_kurt = kurtosis(data)

    return data_mean, data_std, data_skew, data_kurt

def calc_spectral_ratios(signal, fs, f_low=5, f_med=25, f_high=70):
    """
    Return the power ratio contained in 3 bands.
    0 - LF, LF - MF, and MF - HF.

    """
    # Calculate power spectrum using periodogram
    f, pxx = periodogram(signal, fs)

    # Relative areas
    if np.where(f > f_low)[0].size:
        a1 = np.sum(pxx[np.where(f > -1)[0][0]:np.where(f > f_low)[0][0]])
        if np.where(f > f_med)[0].size:
            a2 = np.sum(pxx[np.where(f > f_low)[0][0]:np.where(f > f_med)[0][0]])
            if np.where(f > f_high)[0].size:
                a3 = np.sum(pxx[np.where(f > f_med)[0][0]:np.where(f > f_high)[0][0]])
            else:
                a3 = 1 - a1 - a2
        else:
            a2 = 1 - a1
            a3 = 0
    else:
        a1 = 1
        a2 = 0
        a3 = 0

    a_total = a1 + a2 + a3

    # If there is no spectral power. ie. signal is flatline.
    if a_total == 0:
        return 1, 0 ,0

    return a1 / a_total, a2 / a_total, a3 / a_total


def visualize_features(features, n_bins=20):
    """
    Plot a histogram of each column in a dataframe.
    The 'result' column must be present, and is used to
    discern True and False values.

    """
    for feature_name in features.columns[:-1]:
        feature_true = features.loc[features['result'], feature_name].values
        feature_false = features.loc[features['result']==False, feature_name].values

        feature_true = feature_true[~np.isnan(feature_true)]
        feature_false = feature_false[~np.isnan(feature_false)]

        plt.figure(figsize=(16, 8))
        plt.grid(True)

        n, bins, patches = plt.hist(feature_true, n_bins, normed=1, facecolor='r',
                                    alpha=0.9,)
        n, bins, patches = plt.hist(feature_false, n_bins, normed=1, facecolor='b',
                                    alpha=0.75)

        plt.title(feature_name)
        plt.legend(['True Alarms',
                    'False Alarms'])
        plt.show()


def has_tachycardia(qrs_0, qrs_1):
    """
    Use two sets of beat indices to determine whether or not
    tachycardia has occurred. Only return True if it occurs in
    both channels simultaneously.
    """
    if len(qrs_0) < 5 or len(qrs_1) < 5:
        return False

    # Iterate through groups of 5 channel 0 beats
    for qrs_num in range(len(qrs_0) - 4):
        local_beats_0 = qrs_0[qrs_num:qrs_num + 5]
        local_beats_1 = qrs_1[(qrs_1 > local_beats_0[0] - 40) & (qrs_1 < local_beats_0[-1] + 40)]

        # Too few beats
        if len(local_beats_1) < 5:
            return False

        # rr intervals
        rr = [np.diff(b) for b in [local_beats_0, local_beats_1]]

        rr_mean = [np.mean(r) for r in rr]

        # Ventricular beat intervals must be consistent
        allowed_rr_deviation = [np.mean(r) + 2*np.std(r) for r in rr]
        for ch in range(2):
            if (np.min(rr[ch]) < rr_mean[ch] - allowed_rr_deviation[ch]
                    or np.max(rr[ch]) > rr_mean[ch] + allowed_rr_deviation[ch]):
                return False

        if (processing.calc_mean_hr(rr[0], fs=250) > 100
                and processing.calc_mean_hr(rr[1], fs=250) > 100):
            return True


    return False


def calc_ventricular_training_features():
    """
    Calculate frequency features from labelled
    intervals for identifying VT

    For each record, calculate features from the labelled vtach section
    for both ecg signals.

    In addition, take a 15s interval from another
    arbitrary section of the record.
    """
    fs = 250
    vtach_intervals = {
        'v328s':[293, 296.5],
        'v334s':[296.2, 299.5],
        'v348s':[294, 300],
        'v368s':[290, 293],
        'v369l':[296, 300],
        'v404s':[292, 300],
        'v448s':[294, 299],
        'v471l':[298, 300],
        'v522s':[291, 299],
    }

    features = []

    for record_name in vtach_intervals:
        start_sec = int(vtach_intervals[record_name][0])
        stop_sec = int(vtach_intervals[record_name][1])

        # Read record
        signal, fields = wfdb.rdsamp(os.path.join(data_dir, record_name),
                                     sampfrom=start_sec * fs,
                                     sampto=stop_sec * fs, channels=[0,1])

        # Calculate spectral features for both ecg signals
        features.append(list(calc_spectral_ratios(signal[:, 0], fs=250))+[True])
        features.append(list(calc_spectral_ratios(signal[:, 0], fs=250))+[True])

        # Add spectral ratios for another arbitrary segment
        signal, fields = wfdb.rdsamp(os.path.join(data_dir, record_name),
                                     sampfrom=200 * fs,
                                     sampto=215 * fs, channels=[0,1])

        # Calculate spectral features for both ecg signals
        features.append(list(calc_spectral_ratios(signal[:, 0], fs=250))+[False])
        features.append(list(calc_spectral_ratios(signal[:, 0], fs=250))+[False])

    features = pd.DataFrame(features, columns = ['lfp', 'mfp', 'hfp', 'result'])

    return features


ventricular_training_features = calc_ventricular_training_features()
clf_vent = GradientBoostingClassifier()
clf_vent.fit(ventricular_training_features.iloc[:, :-1],
             ventricular_training_features['result'])



def has_ventricular(signal):
    """
    Figure out whether there is ventricular activity in any 4s window in
    either channel.

    Parameters
    ----------
    signal : numpy array
        2d numpy array of 2 channel ecg
    """

    # Inspect with 1s sliding duration
    ventricular = False
    for window_num in range(7):
        # Get the windowed signal
        sig_window = signal[window_num * fs:(4+window_num) * fs, :]
        # Calculate frequency features

        window_features = [list(calc_spectral_ratios(sig_window[:, 0], fs=250)),
                           list(calc_spectral_ratios(sig_window[:, 1], fs=250))]
        window_ventricular = clf_vent.predict(window_features)
        if np.any(window_ventricular):
            ventricular = True
            break

    return ventricular