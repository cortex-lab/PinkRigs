"""
This script contains a set of functions for calculating the selectivity of neurons
to task and stimulus variables.
"""

import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import scipy.ndimage as spimage
import scipy.signal as spsignal
import scipy.stats as sstats
import xarray as xr
import pickle as pkl
from collections import defaultdict
import itertools
import os


def smooth_spikes(spike_train, method='full_gaussian', sigma=2, window_width=50, custom_window=None,
                  custom_smooth_axis=None):
    """
    Smooths spike trains.
    Parameters
    -------------
    spike_train   : option (1): numpy array of shape (num_time_bin, ) to smooth
                    option (2): numpy array of shape (num_time_bin, num_trial)
    method        : method to perform the smoothing
                   'half_gaussian': causal half-gaussian filter
                   'full_gaussian': standard gaussian filter
    sigma : (int)
    window_width : (int)
    custom_window : if not None, then convolve the spike train with the provided window
    TODO: need to polish handling of edge cases for convolution.
    """

    if custom_smooth_axis is not None:
        smooth_axis = custom_smooth_axis
    elif spike_train.ndim == 2:
        smooth_axis = 1
    elif spike_train.ndim == 1:
        smooth_axis = -1
    else:
        raise ValueError('spike_train dimension is ambigious as to how to perform smoothing')

    # scipy ndimage cnovolution methods used here does not work with ints (round's down)
    if type(spike_train[0]) != np.float64:
        spike_train = spike_train.astype(float)

    if custom_window is None:

        if method == 'full_gaussian':

            smoothed_spike_train = spimage.filters.gaussian_filter1d(spike_train, sigma=sigma,
                                                                     axis=smooth_axis)
            # note that there is a slight offset between the np.convolve and the scipy ndimage implementation, mainly due to handling edge cases I think.

        elif method == 'half_gaussian':
            gaussian_window = spsignal.windows.gaussian(M=window_width, std=sigma)

            # note that the mid-point is included (ie. the peak of the gaussian)
            half_gaussian_window = gaussian_window.copy()
            half_gaussian_window[:int((window_width - 1) / 2)] = 0

            # normalise so it sums to 1
            half_gaussian_window = half_gaussian_window / np.sum(half_gaussian_window)

            smoothed_spike_train = spimage.filters.convolve1d(spike_train, weights=half_gaussian_window,
                                                              axis=smooth_axis)
            # TODO: see https://scipy-cookbook.readthedocs.io/ section on convolutino comparisons

        else:
            print('No valid smoothing method specified')

    else:
        # TODO: convolve custom kernel with the spike train.
        smoothed_spike_train = spsignal.convolve(spike_train, custom_window, mode='same')

    return smoothed_spike_train


def spike_trains_to_spike_rates(spike_trains, bin_size=2 / 1000, start_time=0, end_time=3000):
    """

    Parameters
    ----------
    spike_trains
    bin_size
    start_time
    end_time

    Returns
    -------

    """
    bins = np.arange(start_time, end_time, bin_size)
    num_neurons = len(spike_trains)
    spike_matrix = np.zeros((len(bins) - 1, num_neurons))

    for neuron_n, spike_times in enumerate(spike_trains):
        spike_counts, bins_used = np.histogram(spike_times, bins)
        spike_rate = spike_counts / bin_size

        spike_matrix[:, neuron_n] = spike_rate

    return spike_matrix, bins_used


def align_spike_rate(event_times, spike_bin_times, spike_rate, time_before_align=0, time_after_align=0.3):
    """

    Parameters
    ----------
    event_times
    spike_bin_times
    spike_rate
    time_before_align
    time_after_align

    Returns
    -------

    """
    num_neurons = np.shape(spike_rate)[1]
    num_event = len(event_times)
    time_bin_width = np.mean(np.diff(spike_bin_times))
    num_time_bins = int(time_after_align / time_bin_width)
    aligned_spike_matrix = np.zeros((num_neurons, num_event, num_time_bins))

    for n_event, event_time in enumerate(event_times):
        start_time = event_time - time_before_align
        end_time = event_time + time_after_align

        start_time_idx = np.argmin(np.abs(spike_bin_times - start_time))
        end_time_idx = np.argmin(np.abs(spike_bin_times - end_time)) - 1

        aligned_spike_matrix[:, n_event, :] = spike_rate[start_time_idx:end_time_idx, :].T

    return aligned_spike_matrix


def cal_two_cond_max_diff(spike_rate, cond_1_times, cond_2_times, bins_used,
                          time_before_align=0, time_after_align=0.3,
                          ave_method_across_trials='mean'):
    """

    Parameters
    ----------
    spike_rate
    cond_1_times
    cond_2_times
    bins_used
    time_before_align
    time_after_align

    Returns
    -------

    """

    aligned_spike_matrix_1 = align_spike_rate(
        event_times=cond_1_times, spike_bin_times=bins_used, spike_rate=spike_rate,
        time_before_align=time_before_align, time_after_align=time_after_align
    )

    aligned_spike_matrix_2 = align_spike_rate(
        event_times=cond_2_times, spike_bin_times=bins_used, spike_rate=spike_rate,
        time_before_align=time_before_align, time_after_align=time_after_align
    )

    if ave_method_across_trials == 'mean':
        cond_1_ave_across_trials = np.mean(aligned_spike_matrix_1, axis=1)
        cond_2_ave_across_trials = np.mean(aligned_spike_matrix_2, axis=1)
    elif ave_method_across_trials == 'median':
        cond_1_ave_across_trials = np.median(aligned_spike_matrix_1, axis=1)
        cond_2_ave_across_trials = np.median(aligned_spike_matrix_2, axis=1)

    two_cond_diff = cond_1_ave_across_trials - cond_2_ave_across_trials
    two_cond_abs_diff = np.abs(two_cond_diff)

    # for each cell, calculate maximum difference
    max_abs_diff = np.max(two_cond_abs_diff, axis=1)
    max_abs_diff_idx = np.argmax(two_cond_abs_diff, axis=1)
    max_sign = np.zeros(len(max_abs_diff_idx))

    for cell_n, cell_max_diff_idx in enumerate(max_abs_diff_idx):
        max_sign[cell_n] = np.sign(two_cond_diff[cell_n, cell_max_diff_idx])

    return max_abs_diff, max_sign, max_abs_diff_idx


def cal_neural_selectivity(spike_rate, cond_1_times, cond_2_times, bins_used,
                          time_before_align=0, time_after_align=0.3,
                          ave_method_across_trials='mean', num_shuffle=1000):
    """

    Parameters
    ----------
    spike_rate
    cond_1_times
    cond_2_times
    bins_used
    time_before_align
    time_after_align
    ave_method_across_trials
    num_shuffle

    Returns
    -------

    """

    max_abs_diff, max_sign, max_abs_diff_idx = cal_two_cond_max_diff(spike_rate, cond_1_times, cond_2_times,
                                                                     time_before_align=time_before_align,
                                                                     time_after_align=time_after_align,
                                                                     ave_method_across_trials=ave_method_across_trials)

    num_neuron = np.shape(spike_rate)[1]
    shuffled_max_abs_diff = np.zeros((num_shuffle, num_neuron))

    n_cond_1 = len(cond_1_times)
    n_cond_2 = len(cond_2_times)

    for shuffle_n in tqdm(np.arange(num_shuffle)):
        shuffled_event_times = np.random.permutation(np.concatenate([cond_1_times, cond_2_times]))
        shuffled_cond_1_times = np.sort(shuffled_event_times[0:n_cond_1])
        shuffled_cond_2_times = np.sort(shuffled_event_times[n_cond_1:])
        shuffled_max_abs_diff[shuffle_n, :], _, _ = cal_two_cond_max_diff(spike_rate.reshape(-1, 1),
                                                                          shuffled_cond_1_times, shuffled_cond_2_times,
                                                                          time_before_align=time_before_align,
                                                                          time_after_align=time_after_align,
                                                                          ave_method_across_trials=ave_method_across_trials)

    max_abs_diff_percentile = sstats.percentileofscore(max_abs_diff, shuffled_max_abs_diff)

    return max_abs_diff_percentile



