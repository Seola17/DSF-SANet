import sys

import biosppy.signals.tools as st
import numpy as np
from biosppy.signals.ecg import correct_rpeaks, christov_segmenter
from scipy.interpolate import splev, splrep
from tqdm import tqdm


def segment_recording(signal_recording, sampling_rate, segment_second=60):
    segment_size = sampling_rate * segment_second
    num_segment = int(len(signal_recording) / segment_size)
    segment_list = np.array(signal_recording[0:num_segment * segment_size]).reshape(num_segment, segment_size)
    return segment_list


def noise_reduction(signal, sampling_rate, ftype='FIR', band='bandpass', frequency_min=3, frequency_max=45):
    denoised_signal, _, _ = st.filter_signal(signal, ftype=ftype, band=band,
                                             order=int(0.3 * sampling_rate),
                                             frequency=[frequency_min, frequency_max],
                                             sampling_rate=sampling_rate)
    return denoised_signal


def feature_extraction(signal, sampling_rate):
    hr_min = 20
    hr_max = 300
    num_r_peaks_per_minute_min = 40
    num_r_peaks_per_minute_max = 200
    # using christov algorithm to locate R peak
    idx_r_peaks, = christov_segmenter(signal, sampling_rate=sampling_rate)
    idx_r_peaks, = correct_rpeaks(signal, rpeaks=idx_r_peaks, sampling_rate=sampling_rate, tol=0.1)

    # remove abnormal R peak signal
    minute_sampling_duration = len(signal) / sampling_rate / 60
    num_r_peaks_per_minute = len(idx_r_peaks) / minute_sampling_duration
    if not num_r_peaks_per_minute_min <= num_r_peaks_per_minute <= num_r_peaks_per_minute_max:
        raise Exception('physiologically impossible R peak signal')
    # extract RR intervals(RRI) and R peak amplitudes(RPA)
    rri_tm, rri_signal = idx_r_peaks[1:] / float(sampling_rate), np.diff(idx_r_peaks) / float(sampling_rate)
    ampl_tm, ampl_siganl = idx_r_peaks / float(sampling_rate), signal[idx_r_peaks]
    # medfilt for RRI
    # rri_signal = medfilt(rri_signal, kernel_size=3)
    # ampl_siganl = medfilt(ampl_siganl, kernel_size=3)
    hr = 60 / rri_signal
    # Remove physiologically impossible HR signal
    if not np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
        raise Exception('physiologically impossible HR signal')

    return rri_tm, rri_signal, ampl_tm, ampl_siganl


def cubic_interpolation(rri_tm, rri_signal, ampl_tm, ampl_siganl, target_length):
    tm = np.arange(0, target_length / 3, step=(1 / 3.0))
    interp_rri = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
    interp_ampl = splev(tm, splrep(ampl_tm, ampl_siganl, k=3), ext=1)
    interp_rri = interp_rri.reshape(-1, 1)
    interp_ampl = interp_ampl.reshape(-1, 1)
    return np.concatenate([interp_rri, interp_ampl], axis=-1)


scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def preprocess_segment(segment, sampling_rate, target_length):
    denoised_signal = noise_reduction(segment, sampling_rate)
    rri_tm, rri_signal, ampl_tm, ampl_siganl = feature_extraction(denoised_signal, sampling_rate=sampling_rate)
    feature = cubic_interpolation(rri_tm, rri_signal, ampl_tm, ampl_siganl, target_length)
    return feature


def preprocess_recording(signal_recording, sampling_rate, labels, recording_name):
    segment_list = segment_recording(signal_recording, sampling_rate)
    # number of forward segment
    forward = 2
    # number of backward segment
    backward = 2
    X, X_with_adjacent_segment, y, groups = [], [], [], []
    skip = 0
    for idx_segment in tqdm(range(len(labels)), desc=recording_name, file=sys.stdout):
        is_missing_adjacent_segment = (idx_segment - forward < 0) or (idx_segment + backward > len(segment_list) - 1)
        if is_missing_adjacent_segment:
            continue
        if skip > 0:
            skip = skip - 1
            continue
        segment = segment_list[idx_segment]
        with_adjacent_segment_5 = segment_list[(idx_segment - forward):(idx_segment + backward) + 1]
        with_adjacent_segment_5 = with_adjacent_segment_5.flatten()
        try:
            feature_segment = preprocess_segment(segment, sampling_rate, 180)
            feature_with_adjacent_segment_5 = preprocess_segment(with_adjacent_segment_5, sampling_rate, 900)
        except:
            skip = 2
        else:
            X.append(feature_segment)
            X_with_adjacent_segment.append(feature_with_adjacent_segment_5)
            y.append(0. if labels[idx_segment] == 'N' else 1.)
            groups.append(recording_name)
    return X, X_with_adjacent_segment, y, groups
