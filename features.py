# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

#computes the variance of the window
def _compute_variance_features(window):
    return np.var(window, axis = 0)

#computes the maximums of the window
def _compute_max_features(window):
    return np.max(window, axis = 0)

#computes the minimums of the window
def _compute_min_features(window):
    return np.min(window, axis = 0)

#computes the dominant frequency of the window
def _compute_domfreq_features(window):
    magn = [np.linalg.norm(vals[:3]) for vals in window] #calculates magnitude
    fft = np.fft.rfft(magn, axis = 0)
    return fft.astype(float)

#compute the entropy of the window
def _compute_entropy_features(window):
    magn = [np.linalg.norm(vals[:3]) for vals in window] #calculates magnitude
    hist = np.histogram(magn)
    data = hist[0]/hist[0].sum()
    entropy = -(data * np.ma.log2(np.abs(data))).sum()
    return [entropy]

#returns number of peaks in the window
def _compute_peaks_features(window):
    magn = [np.linalg.norm(vals[:3]) for vals in window] #calculates magnitude
    mean = np.mean(magn)
    peaks = find_peaks(magn, height = mean)[0] #add additional parameters later
    return [len(peaks)]

#computes the mean difference in amplitudes of consecutive samples
def _compute_mean_amplitudes(window):
    magn = [np.linalg.norm(vals[:3]) for vals in window]
    diffs = []
    for i in range(len(magn)-1):
        v = abs(magn[i] - magn[i+1])
        diffs.append(v)
    return [np.mean(diffs)]

#computes the absolute mean of each x, y, z signals
def _compute_absolute_mean(window):
    abso = np.absolute(window)
    return np.mean(abso, axis = 0)

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []
    
    #call functions to compute other features. Append the features to x and the names of these features to feature_names
    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    x.append(_compute_absolute_mean(window))
    feature_names.append('absolute x_mean')
    feature_names.append('absolute y_mean')
    feature_names.append('absolute z_mean')

    x.append(_compute_variance_features(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")

    x.append(_compute_max_features(window))
    feature_names.append("x_max")
    feature_names.append("y_max")
    feature_names.append("z_max")

    x.append(_compute_min_features(window))
    feature_names.append("x_min")
    feature_names.append("y_min")
    feature_names.append("z_min")

    domfreq = _compute_domfreq_features(window)
    x.append(domfreq)
    for i in range(len(domfreq)):
        feature_names.append("dominant frequency")

    x.append(_compute_entropy_features(window))
    feature_names.append("entropy")

    x.append(_compute_peaks_features(window))
    feature_names.append("peaks")

    x.append(_compute_mean_amplitudes(window))
    feature_names.append('mean of amplitudes')

    feature_vector = np.concatenate(x, axis=0)  #convert the list of features to a single 1-dimensional vector
    
    return feature_names, feature_vector