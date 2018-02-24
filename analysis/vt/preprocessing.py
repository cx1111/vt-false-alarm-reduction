"""
Module for preprocessing signals
"""
import numpy as np
from scipy import interpolate
from scipy.stats import mode
from scipy.signal import butter, filtfilt


def fill_missing(sig):
    """
    Fill missing values of a signal by interpolating between
    present samples, and extending the earliest/latest values
    forwards/backwards.
    """
    if sig.ndim == 2:
        clean_sig = np.empty([sig.shape[0], sig.shape[1]])
        for ch in range(sig.shape[1]):
            clean_sig[:, ch] = fill_missing(sig=sig[:, ch])
        return clean_sig

    sig_len = len(sig)
    invalid_inds = np.where(np.isnan(sig))[0]

    n_invalid = invalid_inds.size

    # Return flatline for completely empty signal
    if n_invalid == sig_len:
        return np.zeros(sig_len)

    if n_invalid:
        valid_inds = np.where(~np.isnan(sig))[0]
        valid_samps = sig[valid_inds]
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        f = interpolate.interp1d(valid_inds, valid_samps)

        clean_sig = sig.copy()

        # Set samples on sides of first and last valid samples
        # to their values.
        if valid_inds[0] != 0:
            clean_sig[:valid_inds[0]] = sig[valid_inds[0]]
        if valid_inds[-1] != sig_len - 1:
            clean_sig[valid_inds[-1] + 1:] = sig[valid_inds[-1]]

        invalid_inds = np.where(np.isnan(sig))[0]
        invalid_inds = invalid_inds[(invalid_inds > valid_inds[0]) & (invalid_inds < valid_inds[-1])]
        # Interpolate between existing samples.
        clean_sig[invalid_inds] = f(invalid_inds)
    else:
        clean_sig = sig

    return clean_sig



def get_missing_prop(sig):
    """
    Get the proportion of missing values from the signal
    """
    if sig.ndim == 2:
        return [get_missing_prop(sig[:, ch]) for ch in range(sig.shape[1])]

    sig_len = len(sig)
    nan_locs = np.where(np.isnan(sig))[0]
    return nan_locs.size

def is_missing(sig, missing_thresh=0.4):
    """
    Determine whether a signal has too many missing values.
    Returns True if the ratio of nans exceeds missing_thresh.
    """
    if sig.ndim == 2:
        return [is_missing(sig[:, ch], missing_thresh) for ch in range(sig.shape[1])]

    sig_len = len(sig)
    missing_prop = get_missing_prop(sig)

    if missing_prop / sig_len > missing_thresh:
        return True
    else:
        return False


def get_mode_prop(sig):
    """
    Get the proportion of samples of the signal equal
    to the mode
    """
    if sig.ndim == 2:
        return [get_mode_prop(sig[:, ch]) for ch in range(sig.shape[1])]

    return mode(sig).count[0] / len(sig)

def is_flatline(sig, mode_thresh=0.8):
    """
    Determine whether a signal is flatline
    by inspecting the proportion of samples that
    match the mode, or are invalid
    """
    if sig.ndim == 2:
        return [is_flatline(sig[:, ch], mode_thresh) for ch in range(sig.shape[1])]

    mode_prop = get_mode_prop(sig)

    if mode_prop / len(sig) > mode_thresh:
        return True
    else:
        return False

def get_edge_prop(sig, n_bins=8):
    """
    Get the proportion of the signal in the max and min bins

    """
    if sig.ndim == 2:
        return [get_edge_prop(sig[:, ch], n_bins) for ch in range(sig.shape[1])]

    # get rid of nans
    sig = sig[~np.isnan(sig)]
    if sig.size == 0:
        return 0

    # Bin the data. Get proportion of values in high and low bins
    freq, bin_edges = np.histogram(sig, bins=n_bins)
    edge_prop = (freq[0] + freq[-1]) / np.sum(freq)

    return edge_prop

def get_consec_prop(sig):
    """
    Get the proportion of the signal that shares the same
    value as its previous sample.

    """
    if sig.ndim == 2:
        return [get_consec_prop(sig[:, ch]) for ch in range(sig.shape[1])]

    n_consec = len(np.where(np.diff(sig)==0)[0])
    consec_prop = n_consec / len(sig)
    return consec_prop

def is_saturated(sig, consec_thresh=0.01):
    """
    Determine whether or not a signal is saturated, depending
    on whether the proportion of consecutive samples
    crosses the threshold.
    """
    if sig.ndim == 2:
        return [is_saturated(sig[:, ch], edge_thresh) for ch in range(sig.shape[1])]

    if get_consec_prop(sig) > consec_thresh:
        return True
    else:
        return False


def is_valid(sig, missing_thresh=0.4, mode_thresh=0.8, consec_thresh=0.01):
    """
    Determine whether or not a signal segment is valid.
    It is only valid if it is none of the following:
    - flatline
    - saturated
    - has too many missing values

    """
    if sig.ndim == 2:
        return [is_valid(sig[:, ch], missing_thresh, mode_thresh, consec_thresh) for ch in range(sig.shape[1])]

    if (is_missing(sig, missing_thresh=missing_thresh)
        or is_flatline(sig, mode_thresh=mode_thresh)
        or is_saturated(sig, consec_thresh=consec_thresh)):
        return False
    else:
        return True


def bandpass(sig, fs=250, f_low=0.5, f_high=40, order=4):
    """
    Bandpass filter the signal
    """
    if sig.ndim ==2:
        sig_filt = np.zeros(sig.shape)
        for ch in range(sig.shape[1]):
            sig_filt[:, ch] = bandpass(sig[:, ch], fs, f_low, f_high, order)
        return sig_filt

    f_nyq = 0.5 * fs
    wlow = f_low / f_nyq
    whigh = f_high / f_nyq
    b, a = butter(order, [wlow, whigh], btype='band')
    sig_filt = filtfilt(b, a, sig, axis=0)

    return sig_filt


def normalize(sig):
    """
    Normalize a signal to zero mean unit std
    """
    if np.ptp(sig) == 0:
        return sig

    return (sig - np.average(sig)) / np.std(sig)
