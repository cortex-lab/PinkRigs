# general util
import os
import glob
import numpy as np

# Machine learning / statistics
from sklearn.kernel_ridge import KernelRidge
import sklearn.linear_model as sklinear
import sklearn.metrics as sklmetrics
import sklearn.pipeline as sklpipe
import sklearn.model_selection as sklselection
import sklearn.dummy as skldummy

# Data processing
import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes

# general linear algebra tools
import scipy.linalg as slinalg
import scipy.sparse as ssparse

# Storing data
import collections
import bunch
import pandas as pd
import xarray as xr
import pickle as pkl

# Debugging
import pdb

def spike_df_to_xarray(spike_df, time_bin_width=0.005, bin_start=None, bin_end=None,
                       implementation='loop', meta_dict=None, include_cell_w_no_spike=False,
                       neuron_id=None):
    """
    Converts spike times in spike_df to spike rates, represented as an xarray object.
    Xarray objects can easily be converted to numpy arrays.
    Parameters
    -----------
    spike_df: pandas dataframe
        dataframe with the spike times of each neuron
        the dataframe should contain two columns:
            'spikeTime' - time of the spike in seconds
            'cellId' - the identity of the cell (represented with integers or strings)
    time_bin_width : float
        length (seconds) to bin spikes
    bin_start : float
        when to start binning (in seconds)
        (if None, then start from first spike)
    bin_end : float
        when to stop binning (if None, then end at last spike)
    include_cell_w_no_spike : bool
        whether to include cells with no spikes
    neuron_id : list
        list of unique neuron ids

    Returns
    --------
    binned_spike_ds : xarray dataset
        dataset with dimensions (cell, bin_start)
        and attributes time_bin_width and time_bins
    # TODO: move this to some general spike util script
    """

    if bin_start is None:
        bin_start = min(spike_df['spikeTime'])
    if bin_end is None:
        bin_end = max(spike_df['spikeTime'])

    time_bins = np.arange(bin_start, bin_end, time_bin_width)

    if implementation == 'loop':
        # Naive for loop implementation
        binned_spike_dict = dict()
        if include_cell_w_no_spike:
            unique_cells = neuron_id
        else:
            unique_cells = np.unique(spike_df['cellId'])

        num_cell = len(unique_cells)
        binned_spike_mat = np.zeros(shape=(num_cell, len(time_bins) - 1))

        for n_cell, cell_id in enumerate(unique_cells):
            spike_times = spike_df.loc[
                spike_df['cellId'] == cell_id]['spikeTime']

            # only run the inter-spike-interval test if there is more than
            # one spike
            if len(spike_times) > 1:
                # make sure spike times are monotonically increasing
                assert np.min(np.diff(spike_times)) >= 0
            # binned_spike_dict[cell_id], _ = np.histogram(spike_times,
            #                                          bins=time_bins)
            binned_spike_mat[n_cell, :], _ = np.histogram(spike_times,
                                                          bins=time_bins)
            # Note: histogram returns a vector of zeros when passed an empty
            # pandas series, so no worries there

    binned_spike_ds = xr.Dataset(
        data_vars={
            'spike_rate': (('cell', 'bin_start'), binned_spike_mat)
        },
        coords={
            'bin_start': time_bins[:-1],
            'cell': unique_cells
        })

    binned_spike_ds.attrs['time_bin_length'] = time_bin_width
    binned_spike_ds.attrs['time_bins'] = time_bins

    return binned_spike_ds


def get_event_times(behave_df, event_name, rounding=None):
    """
    Extract time of event of interest into a vector.
    Parameters
    ----------
    behave_df : pandas dataframe
        table with information about the experiment times and trial conditions
    event_name : str
        name of the event to extract the event times
    rounding : int
        decimal place to round the event time (in seconds) to
        if None, then no rounding occurs
    """

    if 'StimPeriodOnOff' in event_name:
        event_times = behave_df[event_name].apply(lambda x: x[0]).values
        event_times = event_times[~np.isnan(event_times)]
    elif ('visLeft' == event_name) or ('visRight' == event_name):
        event_times = behave_df[event_name].dropna().apply(lambda x: x[0]).values
        # event_times = event_times[~np.isnan(event_times)]
    elif event_name == 'visLeftOne':
        # first subset vis left trials (not sure what the above code is doing...)
        event_times = behave_df.loc[behave_df['visDiff'] < 0]['visStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'visRightOne':
        event_times = behave_df.loc[behave_df['visDiff'] > 0]['visStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'audLeftOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['audDiff'] < 0)]['stimOnTime'].values
        # ['audStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'audRightOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['audDiff'] > 0)]['stimOnTime'].values
        # ['audStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'audOn':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff'])
        ]['stimOnTime'].values
    elif event_name == 'audCenterOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['audDiff'] == 0)]['audStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'visLeftAudRightOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] < 0) &
            (behave_df['audDiff'] > 0)
            ]['stimOnTime'].values

    elif event_name == 'visLeftAudLeftOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] < 0) &
            (behave_df['audDiff'] < 0)
            ]['stimOnTime'].values
    elif event_name == 'visRightAudRightOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] > 0) &
            (behave_df['audDiff'] > 0)
            ]['stimOnTime'].values
    elif event_name == 'visRightAudLeftOne':
        event_times = behave_df.loc[
            (np.isfinite(behave_df['audDiff'])) &
            (behave_df['visDiff'] > 0) &
            (behave_df['audDiff'] < 0)
            ]['stimOnTime'].values
    elif event_name == 'visOffAudLeft':
        event_times = behave_df.loc[
            (behave_df['visDiff'] == 0) &
            (behave_df['audDiff'] < 0)
            ]['stimOnTime'].values
    elif event_name == 'visOffAudRight':
        event_times = behave_df.loc[
            (np.isfinite(behave_df['audDiff'])) &
            (behave_df['visDiff'] == 0) &
            (behave_df['audDiff'] > 0)
            ]['stimOnTime'].values
    elif event_name == 'visLeftAudOff':
        event_times = behave_df.loc[
            (behave_df['visDiff'] < 0) &
            (~np.isfinite(behave_df['audDiff']))
            ]['stimOnTime'].values
    elif event_name == 'visRightAudOff':
        event_times = behave_df.loc[
            (behave_df['visDiff'] > 0) &
            (~np.isfinite(behave_df['audDiff']))
            ]['stimOnTime'].values
    elif event_name == 'visLeft0p8One':
        event_times = behave_df.loc[behave_df['visDiff'] == -0.8]['stimOnTime'].values
        # ['visStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'visRight0p8One':
        event_times = behave_df.loc[behave_df['visDiff'] == 0.8]['stimOnTime'].values
        # ['visStimPeriodOnOff'].apply(lambda x: x[0]).values
    elif event_name == 'visLeft0p8AudRightOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] == -0.8) &
            (behave_df['audDiff'] > 0)
            ]['stimOnTime'].values
    elif event_name == 'visLeft0p8AudLeftOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] == -0.8) &
            (behave_df['audDiff'] < 0)
            ]['stimOnTime'].values
    elif event_name == 'visRight0p8AudRightOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] == 0.8) &
            (behave_df['audDiff'] > 0)
            ]['stimOnTime'].values
    elif event_name == 'visRight0p8AudLeftOne':
        event_times = behave_df.loc[
            np.isfinite(behave_df['audDiff']) &
            (behave_df['visDiff'] == 0.8) &
            (behave_df['audDiff'] < 0)
            ]['stimOnTime'].values
    elif event_name == 'visLeft0p8AudOff':
        event_times = behave_df.loc[
            (behave_df['visDiff'] == -0.8) &
            (~np.isfinite(behave_df['audDiff']))
            ]['stimOnTime'].values
    elif event_name == 'visRight0p8AudOff':
        event_times = behave_df.loc[
            (behave_df['visDiff'] == 0.8) &
            (~np.isfinite(behave_df['audDiff']))
            ]['stimOnTime'].values
    elif event_name == 'wheelMoveTime':
        event_times = behave_df['wheelMoveTime'].values
        event_times = event_times[~np.isnan(event_times)]
    elif event_name == 'chooseLeft':
        event_times = behave_df.loc[behave_df['choiceThreshDir'] == 1]['wheelMoveTime'].values
        event_times = event_times[~np.isnan(event_times)]
    elif event_name == 'chooseRight':
        event_times = behave_df.loc[behave_df['choiceThreshDir'] == 2]['wheelMoveTime'].values
        event_times = event_times[~np.isnan(event_times)]
    elif event_name == 'visSign':
        event_times = np.array([x[0] for x in behave_df['visStimPeriodOnOff']])
        event_times = event_times[~np.isnan(event_times)]
    elif event_name == 'visSignTimesAudSign':
        event_times = behave_df['stimOnTime'].values
        event_times = event_times[~np.isnan(event_times)]
    else:
        event_times = behave_df[event_name].values
        event_times = event_times[~np.isnan(event_times)]

    assert event_times is not None

    if rounding is not None:
        event_times = np.round(event_times, rounding)

    return event_times


def get_diag_value_vector(behave_df, event_name):
    """
    Obtain the value to fill the diagonal of the toeplitz matrix of each trial.
    This is usually via some rule mapping from left/right to some values.
    Note that the different diagonal values should sum to 1, eg. left=-1, right=+1, no-go:0

    Parameters
    ----------
    behave_df : pandas dataframe
    event_name : str
    """

    if event_name == 'audDiff':
        assert event_name in behave_df.columns, print('Error: event_name not contained in behave_df')
        event_values = behave_df[event_name].values
        # print('Reminder that aud off will also have a value of 0, '
        #       'nto sure if this will work...')
        # event_values[event_values == np.inf] = 0

        assert len(np.where(event_values == np.Inf)[0]) == 0, \
            print('Error: no audio condition not supported. Remove the Infs')

        # -60 mapped to -1  : audio left
        #  0 mapped to 0    : audio center
        # +60 mpapped to +1 : audio right
        diag_value_vector = np.sign(event_values)

    elif (event_name == 'responseMade') or (event_name == 'choiceThreshDir'):
        assert event_name in behave_df.columns, print('Error: event_name not contained in behave_df')
        event_values = behave_df[event_name].values
        map_dict = {1: -1,  # 1 mapped to -1  : left choice
                    2: 1,  # 2 mapped to +1  : right choice
                    0: 0}  # 0 mapped to  0  : no-go
        diag_value_vector = np.vectorize(map_dict.get)(event_values)

        # remove no goes, because they are won't register as events in movementTimes
        diag_value_vector = diag_value_vector[diag_value_vector != 0]

    elif event_name == 'visSign':
        event_values = behave_df['visDiff'].values
        # [-0.8, -0.4, -0.2] mapped to -1  : visua left
        #  0 mapped to 0    : no visual stimulus
        # [-0.8, -0.4, -0.2] mpapped to +1 : audio right
        diag_value_vector = np.sign(event_values)
    elif event_name == 'visSignTimesAudSign':
        # This is the sign of aud and visual signal multiplied
        # Aud left Vis left --> 1
        # Aud right vis right --> 1
        # Aud left vis right --> -1
        # Aud right vis left --> -1
        vis_values = behave_df['visDiff'].values
        aud_values = behave_df['audDiff'].values
        diag_value_vector = np.sign(vis_values) * np.sign(aud_values)

    return diag_value_vector


def make_event_toeplitz(event_time_vector, event_bins, feat_mat_type='toeplitz',
                        diag_value_vector=None):
    """
    Obtains feature matrix for doing regression.
    INPUT
    -----------
    event_time_vector : (numpy ndarrau)
        binary vector where 1 correspond to the time bin that the event occured
    event_bins        : (numpy ndarray)
        time bins before and after the event that we want to model neural activity
        this always include the time bin '0' corresponding to the onset of the event.
    diag_value_vector : (numpy ndarray)
        vector of the same shape as event_time_vector, to specify the value to fill the diagonal
        for each event, defaults to all 1s.
        (this is often used for (-1, 0, 1) fill values for choice kernels)
    OUTPUT
    ----------
    full_event_toeplitz : (numpy ndarray)
    """

    if feat_mat_type == 'toeplitz':
        # The construction of our toeplitz matrix from a single vector in the middle of the matrix
        # is a bit unconventional, and therefore I have split this to a task wehre we construct
        # two toeplitz matrix: one based on the original vector and one based on the vector flipped
        # and concatenate them.

        post_event_toeplitz = slinalg.toeplitz(c=event_time_vector,
                                               r=np.zeros(shape=(1, len(event_time_vector)))
                                               )
        post_event_toeplitz_truncated = post_event_toeplitz[:, 0:np.sum(event_bins >= 0)]

        pre_event_toeplitz = slinalg.toeplitz(c=np.flip(event_time_vector),
                                              r=np.zeros(shape=(1, len(event_time_vector)))
                                              )
        pre_event_toeplitz_truncated = pre_event_toeplitz[:, 0:np.sum(event_bins < 0)]
        pre_event_toeplitz_flipped = np.flip(pre_event_toeplitz_truncated, axis=(0, 1))

        full_event_toeplitz = np.concatenate([pre_event_toeplitz_flipped[:, :-1],
                                              post_event_toeplitz_truncated], axis=1)

        # feature_matrix = full_event_toeplitz

        # make sure there are no events whose diagonal is being cut off prematurely
        # (ie. if event happens at time point 7, and the total duration of the trial is 8,
        # and the neural time window is 5, then the diagonal of that event will be cut off as time point 8)
        # assert np.sum(full_event_toeplitz[:, -1]) == np.sum(event_time_vector)

    elif feat_mat_type == 'fill-diagonal-toeplitz':
        # this implementation is far more memory efficient than using toeplitz
        # since it does not have to make a square (useful for tall matrices)

        # the underlying logic is the same; we still need to make two matrices
        # actually there is no need if I can infer the first vector
        # I guess that can be done by some form of sliding...
        # but then I also need to know the first row... so overall seems
        # complicated...

        event_bin_mid_point = np.where(event_bins == 0)[0]
        post_event_toeplitz = np.zeros(shape=(len(event_time_vector),
                                              len(np.where(event_bins >= 0)[0])
                                              ))
        # we include the midpoint in the post-event toeplitz
        event_time_indices = np.where(event_time_vector == 1)[0]

        if diag_value_vector is None:
            diag_value_vector = np.full(shape=np.shape(event_time_indices), fill_value=1)

        assert np.shape(event_time_indices)[0] == np.shape(diag_value_vector)[0], \
            print('Error: event time index has %.f elements, but diagonal value vector has %.f elements'
                  % (np.shape(event_time_indices)[0], np.shape(diag_value_vector)[0]))

        # Still requires for loop... which annoys me.
        for diag_index, diag_value in zip(event_time_indices, diag_value_vector):
            np.fill_diagonal(post_event_toeplitz[diag_index:, :], diag_value)

        pre_event_toeplitz = np.zeros(shape=(len(event_time_vector),
                                             len(np.where(event_bins <= 0)[0])
                                             ))

        flipped_event_time_indices = np.where(np.flip(event_time_vector) == 1)[0]
        flipped_diag_value_vector = np.flip(diag_value_vector)

        for diag_index, diag_value in zip(flipped_event_time_indices, flipped_diag_value_vector):
            np.fill_diagonal(pre_event_toeplitz[diag_index:, :], diag_value)

        pre_event_toeplitz_flipped = np.flip(pre_event_toeplitz,
                                             axis=(0, 1))
        # remove middle column (associated with onset of event
        pre_event_toeplitz_truncated = pre_event_toeplitz_flipped[:, :-1]

        full_event_toeplitz = np.concatenate([pre_event_toeplitz_truncated,
                                              post_event_toeplitz],
                                             axis=1)

        if np.sum(full_event_toeplitz[:, -1]) != np.sum(event_time_vector):
            print('Warning: there are events that got truncated in '
                  'peri-stimulus time bins')
        # assert np.sum(full_event_toeplitz[:, -1]) == np.sum(event_time_vector)

    return full_event_toeplitz


def test_toeplitz(event_times=np.array([3, 5, 8]), event_bin_start=-3,
                  event_bin_end=3, time_bin_length=1, spike_start=1,
                  spike_end=11, feat_mat_type='toeplitz'):
    """
    Testing the code which generates a toeplitz matrix using the middle column.
    This utilise the scipy toeplitz function, but requires some modification
    because it can only generate the toeplitz from the first column (or the first row)
    """

    event_bins = np.arange(event_bin_start, event_bin_end + 1, time_bin_length)
    spike_time_bins = np.arange(spike_start, spike_end + 1, time_bin_length)

    event_time_vector, _ = np.histogram(event_times, spike_time_bins)

    toeplitz_matrix = make_event_toeplitz(event_time_vector=event_time_vector,
                                          event_bins=event_bins,
                                          feat_mat_type=feat_mat_type)

    print('Events occured at: ')
    # Plot the result to make sure it makes sense
    fig, ax = vizregression.plot_toeplitz(fig=None, ax=None,
                                          toeplitz_matrix=toeplitz_matrix,
                                          global_time=spike_time_bins,
                                          peri_event_time=event_bins)
    fig.show()

    return fig, ax


def make_feature_matrix(behaviour_df, event_names, event_start_ends,
                        spike_bins, bin_width, custom_feature_names=None,
                        feat_mat_type='fill-diagonal-toeplitz',
                        movement_time_name='movementTimes', return_event_times=False):
    """
    Generates as toeplitz for each trial, then combine all trials together.
    Also generates a toeplitz for each event type, and combine them together.
    Also adds a column of ones in the beginning (for the bias b_0 term)

    Parameters
    ------------
    behaviour_df (pandas dataframe)


    event_start_ends (numpy ndarray)
        matrix of shape (n_event, 2) each row correspond to the start and end times (seconds) which
        you want to bin the peri-event time.


    Returns
    -----------
    feature_matrix : (numpy ndarray)
        feature matrix of shape (num_sample, num_features)
    feature_column_dict: dict
        key: feature name
        value: indices containing the column assocaited with that feature
        the intercept column will have the feature name 'intercept'
        If custom_feature_names is None, then event_name will be used as the feature name
        otherwise this will iterate through custom_feature_names in order.
    event_times_dict : (dict, optional output)
    """

    # Events where the diagonal values are not just either 0 or 1
    events_w_special_diagonals = ['audDiff', 'responseMade', 'visSign', 'visSignTimesAudSign']

    if event_start_ends.ndim == 1:
        # if there is only one row, then expand to have row dimension of 1
        event_start_ends = np.expand_dims(event_start_ends, axis=0)

    event_toeplitz_list = list()
    feature_column_dict = dict()

    feature_column_start_idx = 1

    if len(event_names) == 0:
        print('No events provided, returning vector of 1 for use as baseline model.')
        feature_matrix = np.full(shape=(len(spike_bins) - 1, 1), fill_value=1)
        feature_column_dict = {'baseline': 1}
        if return_event_times:
            return feature_matrix, feature_column_dict, None
        else:
            return feature_matrix, feature_column_dict

    event_times_dict = dict()

    for n_event, event_name in enumerate(event_names):
        event_bin_start, event_bin_end = event_start_ends[n_event, :]

        # some special events have event_times attached to something else
        # TODO: will be more principled to have a dictionary / tuple to deal with this.
        if event_name == 'audDiff':
            event_times = get_event_times(behaviour_df, 'audStimPeriodOnOff')
        elif event_name == 'responseMade':
            event_times = get_event_times(behaviour_df, movement_time_name)
        elif event_name == 'choiceThreshDir':
            event_times = get_event_times(behaviour_df, movement_time_name)
        else:
            event_times = get_event_times(behaviour_df, event_name)

        event_time_vector, _ = np.histogram(event_times, spike_bins)
        event_bins = np.arange(event_bin_start, event_bin_end, bin_width)
        round_decimal_place = 5  # there seems to be some weird issue with numpy arange for floats
        # eg. see: https://github.com/numpy/numpy/issues/5808

        event_bins = np.round(event_bins, round_decimal_place)

        if event_name in events_w_special_diagonals:
            diag_value_vector = get_diag_value_vector(behave_df=behaviour_df, event_name=event_name)
        else:
            diag_value_vector = None

        try:
            event_toeplitz = make_event_toeplitz(event_time_vector,
                                                 event_bins, feat_mat_type=feat_mat_type,
                                                 diag_value_vector=diag_value_vector)
        except:
            pdb.set_trace()

        event_toeplitz_list.append(event_toeplitz)

        # update dictionary with the columns associated with this feature
        feature_column_end_idx = feature_column_start_idx + len(event_bins)
        if custom_feature_names is None:
            feature_column_dict[event_name] = np.arange(feature_column_start_idx, feature_column_end_idx)
        else:
            feature_column_dict[custom_feature_names[n_event]] = np.arange(feature_column_start_idx,
                                                                           feature_column_end_idx)
        feature_column_start_idx = feature_column_end_idx

        # Store the event times to be used for alignment in plotting.
        event_times_dict[event_name] = event_times

    if len(event_toeplitz_list) == 1:
        all_event_toeplitz = event_toeplitz_list[0]
    else:
        all_event_toeplitz = np.concatenate(event_toeplitz_list, axis=1)

    # Add a first column diagonal of 1s
    first_column = np.full(shape=(len(spike_bins) - 1, 1), fill_value=1)
    feature_matrix = np.concatenate([first_column, all_event_toeplitz],
                                    axis=1)

    if return_event_times:
        return feature_matrix, feature_column_dict, event_times_dict
    else:
        return feature_matrix, feature_column_dict


def data_to_X_Y_matrix(behave_df, neuron_df, spike_df=None, subject=None, experiment=None, target_cell_loc=None,
                       event_names=['visLeftOne', 'visRightOne', 'audLeftOne', 'audRightOne'],
                       event_start_ends=np.array(
                           [[-0.05, 0.4],
                            [-0.05, 0.4],
                            [-0.05, 0.4],
                            [-0.05, 0.4]]), bin_width=5 / 1000, smooth_spikes=True, smooth_sigma=50,
                       smooth_window_width=100,
                       subset_stim_cond=None, do_remove_all_zero_rows=False, group_by_trial=True,
                       trial_conds_to_get=None,
                       return_event_times=True, trial_idx_method='arange', divide_by_time_bin=True):
    """
    Go from raw data to feature matrix (basically wrapping make_feature_matrix
    and targets.
    Parameters
    ----------
    ephys_behave_df : (pandas dataframe)
    neuron_df : (pandas dataframe)
    spike_df : (pandas dataframe)
    event_names : (list)
    event_start_ends : (numpy ndarray)
    bin_width : (float)
        time in seconds to bin neural activity
    subset_stim_cond : (list of dict)
        list of dictionary of stimulus condition to subset
    remove_all_zero_rows : (bool)
        whether to remove feature matrix where rows are all zeros (excluding the intercept column)
        this may speed up regression / reduce memory demand depending on the model used and its implementation.
    Returns
    -------

    """

    # Subset data
    if subject is not None:
        behave_df = behave_df.loc[
            behave_df['subjectRef'] == subject
            ]

        neuron_df = neuron_df.loc[
            neuron_df['subjectRef'] == subject
            ]

    if experiment is not None:
        behave_df = behave_df.loc[
            behave_df['expRef'] == experiment
            ]

        neuron_df = neuron_df.loc[
            neuron_df['expRef'] == experiment
            ]

    if target_cell_loc is not None:
        neuron_df = neuron_df.loc[(neuron_df['cellLoc'].isin(target_cell_loc))]

    if subset_stim_cond is not None:
        stim_cond_behave_df_list = list()
        for stim_cond in subset_stim_cond:
            stim_cond_behave_df = behave_df.loc[
                (behave_df['audDiff'] == stim_cond['audDiff']) &
                (behave_df['visDiff'] == stim_cond['visDiff'])
                ]
            stim_cond_behave_df_list.append(stim_cond_behave_df)
        behave_df = pd.concat(stim_cond_behave_df_list)

    neuron_id = neuron_df['cellId'].tolist()
    spike_df = spike_df.loc[spike_df['cellId'].isin(neuron_id)]

    if len(spike_df) == 0:
        print('No spikes found, returning None')
        return None, None, None, None, None, None

    # Make target matrix Y
    target_spike_ds = spike_df_to_xarray(spike_df=spike_df, time_bin_width=bin_width)

    # Smooth spikes
    # TODO: currently code may not work without smoothing.
    if smooth_spikes:
        target_spike_ds['spike_rate'] = (['cell', 'bin_start'], anaspikes.smooth_spikes(
            target_spike_ds['spike_rate'],
            method='half_gaussian',
            sigma=smooth_sigma, window_width=smooth_window_width,
            custom_window=None))

    spike_bins = target_spike_ds.attrs['time_bins']

    Y = target_spike_ds.spike_rate.values.T

    # Convert to spike/s
    if divide_by_time_bin:
        Y = Y / bin_width

    # Make feature matrix
    if (type(event_names) is dict) & (type(event_start_ends) is dict):
        print('Generating multiple feature matrices')
        X = dict()
        feature_column_dict = dict()
        timing_info = dict()

        for feature_set_name, event_names_set in event_names.items():

            event_start_end_set = event_start_ends[feature_set_name]

            X_set, feature_column_dict_set, event_times_dict = make_feature_matrix(
                behave_df.sort_index(), event_names=event_names_set, event_start_ends=event_start_end_set,
                spike_bins=spike_bins, bin_width=bin_width, custom_feature_names=None,
                feat_mat_type='fill-diagonal-toeplitz', return_event_times=return_event_times)

            if group_by_trial:
                if trial_conds_to_get is None:
                    group_vector = make_trial_vec(behave_df=behave_df,
                                                  spike_ds=target_spike_ds,
                                                  trial_conds_to_get=trial_conds_to_get)
                else:
                    group_vector, trial_conds_df = make_trial_vec(behave_df=behave_df,
                                                                  spike_ds=target_spike_ds,
                                                                  trial_conds_to_get=trial_conds_to_get,
                                                                  trial_idx_method=trial_idx_method)


            else:
                group_vector = None

            if do_remove_all_zero_rows:
                print('Removing rows with all zeros (not counting the itnercept')
                X_set, Y_set, trial_vector = remove_all_zero_rows(X=X_set, Y=Y,
                                                                  group_vector=group_vector,
                                                                  intercept_included=True)
            else:
                trial_vector = group_vector

                #  Activity not in trial are separately grouped (pseudo-trials)
                # TODO: group each consecutive NaN into a different "trial"
                trial_vector[np.isnan(trial_vector)] = np.nanmax(trial_vector) + 1

            X[feature_set_name] = X_set
            feature_column_dict[feature_set_name] = feature_column_dict_set

            # Add timing info
            timing_info[feature_set_name] = dict()
            timing_info[feature_set_name]['spike_bins'] = spike_bins
            timing_info[feature_set_name]['event_times'] = event_times_dict

    else:
        X, feature_column_dict, event_times_dict = make_feature_matrix(
            behave_df, event_names=event_names, event_start_ends=event_start_ends,
            spike_bins=spike_bins, bin_width=bin_width, custom_feature_names=None,
            feat_mat_type='fill-diagonal-toeplitz', return_event_times=return_event_times)

        if group_by_trial:
            group_vector, trial_conds_df = make_trial_vec(behave_df=behave_df,
                                                          spike_ds=target_spike_ds,
                                                          trial_conds_to_get=trial_conds_to_get)
        else:
            group_vector = None

        if do_remove_all_zero_rows:
            X, Y, trial_vector = remove_all_zero_rows(X=X, Y=Y, group_vector=group_vector,
                                                      intercept_included=True)
        else:
            trial_vector = group_vector

        timing_info = dict()
        timing_info['spike_bins'] = spike_bins
        timing_info['event_times'] = event_times_dict

    return X, feature_column_dict, Y, trial_vector, trial_conds_df, timing_info


def make_target_matrix(spike_df, bin_width=0.05):
    print('Do nothing')

    return target_matrix


def make_aligned_target_matrix(target_spike_ds, behaviour_df, event_name, event_start_ends,
                               bin_width, method='xarray'):
    """
    Makes a target regression matrix Y from dataset with spike activity and event times of interest
    Parameters
    ----------
    target_spike_ds : xarray dataset
    behaviour_df : pandas dataframe
    event_name : str
    event_start_ends : list
        list with two elements, representing the support of the kernel aligned to event onset in seconds
        Eg. [-1, 2] will mean the kernel goes from one second before the event onset to two seconds after the onset
    bin_width : float
        width of spike time binning used in target_spike_ds
    method : str
        whether the input given in target_spike_ds is an xarray dataset ('xarray') or numpy ndarray ('numpy')
    Returns
    -------
    TODO: numpy method not supported yet
    """
    event_bin_start = event_start_ends[0]
    event_bin_end = event_start_ends[1]
    if method == 'numpy':

        event_times = get_event_times(behaviour_df, event_name)

        for e_time in event_times:
            event_time_vector, _ = np.histogram(event_times, spike_bins)
            event_bins = np.arange(event_bin_start, event_bin_end, bin_width)
    elif method == 'xarray':

        event_time = get_event_times(behaviour_df, event_name)
        window_start = event_time + event_bin_start
        window_end = event_time + event_bin_end

        event_bins = list(zip(window_start,
                              window_end))

        event_interval_bins = pd.IntervalIndex.from_tuples(event_bins, closed='both')
        aligned_xarray_tuple = target_spike_ds.groupby_bins('Time', event_interval_bins)
        aligned_xarray_list = [i[1] for i in list(aligned_xarray_tuple)]

    return aligned_target_matrix


def make_trial_vec(behave_df, spike_ds, trial_idx_method='arange', trial_conds_to_get=None):
    """
    Make vector of the trial of each time point. This is to be used
    in kernel regression to group trials so that the time points within
    each trial are not split into testing and training set (this prevents
    learning of autocorrelation within trials)

    Parameters
    ----------
    behave_df : (pandas dataframe)
        dataframe where each row is a trial
        required columns are 'trialStart' and 'trialEnd', in unit of seconds
    spike_ds : (xarray dataset or dataarray)
        spike dataset containing 'bin_start' as the time dimension (in seconds)
    trial_idx_method : (str)
        how to label the trials
        'arange' : just the count from 0 to number of rows in behave df, note that this vector
                    may later by subsetted
        'global' : use the index label that is unique to every trial in the entire dataset; it
        is created in the creation of ephys_behave_df, this allows cross-referencing
        to alignment_ds
    Returns
    -------
    trial_vector : (numpy ndarray)
        vector of length equal to number of spike time bins
        nan denotes not a trial (eg. before the first trial or after the last trial)
        trial starts counting at 0
    """

    spike_ds['trial'] = (['bin_start'],
                         np.repeat(np.nan, len(spike_ds.bin_start)))

    bstart = spike_ds['bin_start']

    for n_trial, (trial_idx, trial_df) in enumerate(behave_df.iterrows()):
        trial_start = trial_df['trialStart']
        trial_end = trial_df['trialEnd']

        if trial_idx_method == 'arange':
            spike_ds['trial'].loc[
                dict(bin_start=bstart[(
                        (bstart >= trial_start) & (bstart <= trial_end)
                )])] = n_trial
        elif trial_idx_method == 'global':
            spike_ds['trial'].loc[
                dict(bin_start=bstart[(
                        (bstart >= trial_start) & (bstart <= trial_end)
                )])] = trial_df.name
        else:
            print('WARNING: no valid trial idx method specified')

    trial_vector = spike_ds['trial'].values

    if trial_conds_to_get is None:
        return trial_vector
    else:
        # trials_included = np.sort(np.unique(trial_vector[~np.isnan(trial_vector)]))
        # unique in order of apperance
        trials_included = pd.unique(trial_vector[~np.isnan(trial_vector)])
        trial_conds_to_get = ['audDiff', 'visDiff', 'stimOnTime']
        if trial_idx_method == 'arange':
            # behave_df = behave_df.sort_index()  # sort by trial ordering
            trial_cond_df = behave_df[trial_conds_to_get].iloc[trials_included]
            trial_cond_df['trial_idx'] = trials_included
            # trial_cond_df['trial_idx'] = np.arange(len(trials_included))
        elif trial_idx_method == 'global':
            trial_cond_df = behave_df[trial_conds_to_get].loc[trials_included]
            trial_cond_df['trial_idx'] = trials_included
        return trial_vector, trial_cond_df


def remove_all_zero_rows(X, Y, group_vector=None, intercept_included=True):
    """
    Delete rows of the feature matrix and the target matrix in location
    when there are all zeros in the rows of the feature matrix
    Parameters
    ----------
    X : (numpy ndarray)
        feature matrix
    Y : (numpy ndarray)
        target matrix
    Returns
    -------
    reduced_X : (numpy ndarray)
        X with all-zero row deleted
    reduced_Y : (numpy ndarray)
        Y with rows corresponding to all-zero row in X deleted
    reduced_group_vector : (numpy ndarray)
        1 dimension vector with the groups to remove
    """

    # double check whether X has the intercept included
    if intercept_included:
        if np.shape(X)[1] == 1:
            # only a single feature
            all_zero_rows = np.where(X == 0)[0]
        else:
            all_zero_rows = np.where(~X[:, 1:].any(axis=1))[0]
    else:
        all_zero_rows = np.where(~X.any(axis=1))[0]
        if np.sum(X[:, 0]) == np.shape(X)[0]:
            print('Are you sure the intercept is not included? Because'
                  'the first column seems to be all zeros')

    reduced_X = np.delete(X, all_zero_rows, axis=0)
    reduced_Y = np.delete(Y, all_zero_rows, axis=0)

    if group_vector is not None:
        reduced_group_vector = np.delete(group_vector, all_zero_rows, axis=0)
    else:
        reduced_group_vector = None

    return reduced_X, reduced_Y, reduced_group_vector


class reduce_feature_matrix(object):
    """
    Performs dimensionality reduction on the feature matrix.
    Currently only supports reduced rank regression.

    Parameters
    ----------------
    X : (numpy ndarray)
        feature matrix with shape (num_samples, num_features)
    Y : (numpy ndarray)
        target matrix with shape (num_samples, num_target_variables)
    rank : int
        number of dimensions to keep
    method : str
        how to implement the dimensionality reduction
        'reduced-rank' : perform reduced rank regression using code from  Chris Rayner (2015)
        'reduced-rank-steinmetz': same implementation as found in Stienmetz (2019)
        see: https://bit.ly/34jQf4Z

    # TODO: perhaps take the PB matrix in fit, then trasfnrom will just be PBX
    """

    def __init__(self, rank=5, reg=0, method='reduced-rank'):
        self.rank = rank
        self.reg = reg
        self.method = method
        self.transformer_matrix = None  # needs to be fitted.

    def fit(self, X, Y):

        assert self.method in ['reduced-rank', 'reduced-rank-steinmetz'], print('Unknown method specified.')
        if self.method == 'reduced-rank':
            # weird implementation, but get the same results as Kush's reduced rank code
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXY = np.dot(X.T, Y)
            _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

            W = V[0:self.rank, :].T
            A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, W)).T
            self.transformer_matrix = A.T  # same as B in Steinmetz 2019

            # XA = np.dot(X, A.T)  # same as PB in Steinmetz 2019
            # reduced_X = XA

        elif self.method == 'reduced-rank-steinmetz':

            CYX = np.dot(Y.T, X)
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXXMH = np.sqrt(CXX)
            M = np.dot(CYX, CXXMH)

            U, S, V = np.linalg.svd(M)
            B = np.dot(CXXMH, V)
            # _A = np.dot(U, S)

            reduced_B = B[:, :self.rank]
            self.transformer_matrix = reduced_B
            # reduced_X = np.dot(X, reduced_B)

        return self

    def transform(self, X):

        reduced_X = np.dot(X, self.transformer_matrix)

        return reduced_X

    def fit_transform(self, X, Y):

        if self.method == 'reduced-rank':
            # weird implementation, but get the same results as Kush's reduced rank code
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXY = np.dot(X.T, Y)
            _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

            W = V[0:self.rank, :].T
            A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, W)).T
            XA = np.dot(X, A.T)  # same as PB in Steinmetz 2019
            reduced_X = XA

        elif self.method == 'reduced-rank-steinmetz':

            CYX = np.dot(Y.T, X)
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXXMH = np.sqrt(CXX)
            M = np.dot(CYX, CXXMH)

            U, S, V = np.linalg.svd(M)
            B = np.dot(CXXMH, V)
            # _A = np.dot(U, S)

            reduced_B = B[:, :self.rank]
            reduced_X = np.dot(X, reduced_B)

        return reduced_X

    def get_params(self, deep=True):
        return {'rank': self.rank, 'reg': self.reg}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    Implemented by Chris Rayner (2015): dchrisrayner AT gmail DOT com.
    With some extensions to by Tim Sit to calculate the error.
    Also restructured in a way that fits better with sklearn model objects.
    """

    def __init__(self, rank=3, reg=0, regressor=None, alpha=0, l1_ratio=1):
        self.reg = reg
        self.rank = rank
        self.regressor = regressor
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def fit(self, X, Y):

        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        # _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        matrix_to_do_SVD = np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY))

        if np.isnan(matrix_to_do_SVD).any():
            # TODO: perhaps the better option is to drop those feature columns...
            print('NaNs detected, replacing them with zeros')
            matrix_to_do_SVD[np.isnan(matrix_to_do_SVD)] = 0

        _U, _S, V = np.linalg.svd(matrix_to_do_SVD)
        self.W = V[0:self.rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

        if self.regressor == 'Ridge':
            self.XA = np.dot(X, self.A.T)  # same as PB in Steinmetz 2019
            self.regressor_model = sklinear.Ridge(alpha=self.alpha, fit_intercept=False, solver='auto')
            self.regressor_model.fit(X=self.XA, y=Y)
        elif self.regressor == 'ElasticNet':
            self.XA = np.dot(X, self.A.T)  # same as PB in Steinmetz 2019
            self.regressor_model = sklinear.ElasticNet(fit_intercept=False, alpha=self.alpha,
                                                       l1_ratio=self.l1_ratio)
            self.regressor_model.fit(X=self.XA, y=Y)

        return self

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))

        if self.regressor is None:
            Y_hat = np.dot(X, np.dot(self.A.T, self.W.T))
        else:
            XA = np.dot(X, self.A.T)  # multiply new data with the A (ie. B) matrix that was learnt
            Y_hat = self.regressor_model.predict(XA)

        return Y_hat

    def score(self, X, Y):

        Y_hat = self.predict(X)
        u = np.sum(np.power(Y - Y_hat, 2))  # residual sum of squares
        v = np.sum(np.power(Y - np.mean(Y), 2))  # total sum of squares
        R_2 = 1 - u / v

        return Y, R_2

    def get_params(self, deep=True):
        return {'rank': self.rank, 'reg': self.reg}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def groupKFoldRandom(groups, n=2, seed=None):
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    Built upon this: https://bit.ly/2Pp7TQs
    Parameters
    -----------
    groups : (list or numpy array)
        vector contaning the group associated with each sample
    n      : (int)
        number of cross validation splits, must be 2 or greater
    seed   : (int or None)
        random state of the shuffling
    Returns
    result : list
        list of tuples that contain the train and test indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


def fit_kernel_regression(X, Y, method='KernelRidge', fit_intercept=False, evaluation_method='fit-all', cv_split=10,
                          num_repeats=5,
                          tune_hyper_parameter=True, split_group_vector=None, preprocessing_method=None,
                          rank=10, dev_test_random_seed=None, cv_random_seed=None, test_size=0.2,
                          save_path=None, check_neg_weights=True, time_bins=None, save_X=False):
    """
    Fits kernel regression to neural data.
    Parameters
    -----------
    X :  (numpy ndarray)
         feature matrix (toeplitz-like matrix)
    Y :  (numpy ndarray)
        neural activity (can be one vector or a matrix of shape (numTimePoints, numNeuron)
    evaluation_method : (str)
        How you want to train and evaluate the model
        'fit-all' : fit the entire dataset with the model
        'train-cv-test' : fit to train set, use CV set to tune the hyperparameter, then evaluate the best model
        on the test set.
    dev_test_random_seed : (int)
        random seed to split the development and testing set
    test_size : (float)
        test set size
        value has to be between 0 and 1 (non-inclusive), usually sensible values ranges
        from 0.1 to 0.5
    split_group_vector : (numpy ndarray)
        group that is used to generate cross validation / test train splits.
        most common use case is to specify the trial number of each time bin using this vector,
        so that when you do test-train and cv splits you split by trial number (instead of randomly)
        this is useful when time bins are not independent of each other (ie. there is autocorrelation)
    save_X : (booL)
        whether to save the feature matrices as well (for double checking things)

    Returns
    ----------
    fit_results : (Bunch (dict-like) object)
        dictionary with the following keys, but this depends on which evaluation_method is used.
        test_raw_r2_score :  r2_score in the test set, this is 1 - [(y_true - y_pred)^2/(y_true - y_true_mean)^2]
                             see: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        test_var_weighted_r2_score : variance-weighted r_2 score in the test set
        test_explained_variance : variance explained in test set
        y_test_prediction : prediction of the test set
        y_test : actual values in the test set
        dev_model : model used in the development set (train set and validation set)
        test_model : model fitted to the test set (this is to get the maximum explainable variance in the test set)
        y_dev : actual values in the development set
        y_dev_prediction : predicted (fitted) values in the development set
        dev_explained_variance : explained variance in the the development set
        test_set_explainable_variance : possible explainable variance in the test set by fitting a model to it
    """

    # TODO: perhaps the below can be better separated to (1) defining the model and (2) evaluating the model
    fit_results = bunch.Bunch()

    if split_group_vector is not None:
        assert len(split_group_vector) == np.shape(X)[0]

    # List of supported methods that fits into the sklearn workflow
    sklearn_compatible_methods = ['KernelRidge', 'Ridge', 'ReducedRankRegression',
                                  'ReduceThenElasticNetCV', 'ReduceThenRidgeCV',
                                  'nnegElasticNet', 'DummyRegressorMean']
    other_methods = ['ReduceThenRidge', 'ReduceThenElasticNet']

    if preprocessing_method == 'RankReduce':
        print('Performing reduced rank regression as a preprocessing step')
        reduction_transformer = reduce_feature_matrix(reg=0, rank=rank)
        reduction_transformer.fit(X, Y)
        X = reduction_transformer.transform(X)

    if (method in sklearn_compatible_methods) or ('sklearn' in str(type(method))):

        if evaluation_method == 'fit-all':

            if method == 'KernelRidge':
                clf = KernelRidge(alpha=0.0, kernel='linear')
            elif method == 'Ridge':
                # the column of 1s is already included in design matrix, so no need to fit intercept.
                clf = sklinear.Ridge(alpha=0.0, fit_intercept=fit_intercept, solver='auto')
            elif method == 'nnegElasticNet':
                clf = sklinear.ElasticNet(alpha=1.0, fit_intercept=fit_intercept,
                                          positive=True)
            elif method == 'ReducedRankRegression':
                clf = ReducedRankRegressor(rank=50, reg=1.0)
            elif method == 'ReduceThenRidgeCV':
                clf = ReducedRankRegressor(regressor='Ridge', reg=5)
                # parameters to tune are (1) rank (2) alpha for the ridge
            elif method == 'ReduceThenElasticNetCV':
                clf = ReducedRankRegressor(regressor='ElasticNet')
            elif method == 'DummyRegressorMean':
                clf = skldummy.DummyRegressor(strategy='mean')
            else:
                clf = method

            # check that there are no negative weights in the feature matrix
            if check_neg_weights:
                if (X < 0).any():
                    print('Found negative weights in feature matrix.')
                    print('To disable negative weight checking, set check_neg_weights to False')

            fitted_model = clf.fit(X, Y)
            y_predict = clf.predict(X)
            raw_r2_score = sklmetrics.r2_score(Y, y_predict, multioutput='raw_values')
            var_weighted_r2_score = sklmetrics.r2_score(Y, y_predict, multioutput='variance_weighted')

            explained_variance_per_neuron = sklmetrics.explained_variance_score(y_true=Y,
                                                                                y_pred=y_predict,
                                                                                multioutput='raw_values')

            fit_results = dict()
            fit_results['y'] = Y
            fit_results['y_predict'] = y_predict
            fit_results['raw_r2_score'] = raw_r2_score
            fit_results['var_weighted_r2_score'] = var_weighted_r2_score
            fit_results['explained_variance_per_neuron'] = explained_variance_per_neuron
            fit_results['model'] = fitted_model
            fit_results['time_bins'] = time_bins

            # TODO also include the kernels ??? But I guess those are already in y_predict

        elif evaluation_method == 'cv':
            print('Evaluating via repeated k-fold cross validation')
            print('This is a TODO')

        elif evaluation_method == 'train-cv-test':

            if split_group_vector is None:
                X_dev, X_test, y_dev, y_test = sklselection.train_test_split(X, Y, test_size=test_size,
                                                                             random_state=dev_test_random_seed)
            else:
                gss = sklselection.GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=dev_test_random_seed)
                dev_inds, test_inds = next(gss.split(X, Y, groups=split_group_vector))
                X_dev, X_test, y_dev, y_test = X[dev_inds, :], X[test_inds, :], Y[dev_inds], Y[test_inds]
                split_group_vector_dev, split_group_vector_test = split_group_vector[dev_inds], \
                                                                  split_group_vector[test_inds]

                if time_bins is not None:
                    time_bins_dev = time_bins[dev_inds]
                    time_bins_test = time_bins[test_inds]

            # Perform cross validation on the 'development' data
            if method == 'KernelRidge':
                model = KernelRidge(alpha=1.0, kernel='linear')
            elif method == 'Ridge':
                # default model is basically not regularised
                model = sklinear.Ridge(alpha=0.000001, fit_intercept=fit_intercept, solver='auto')
                param_grid = [{'alpha': np.logspace(-4, 4, num=10)}]
                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)
            elif method == 'ReducedRankRegression':
                model = ReducedRankRegressor()
                param_grid = [{'rank': np.linspace(2, 200, 10).astype(int),
                               'reg': np.logspace(-4, 4, num=10)}]

                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)  # TODO: needs to be specified
            elif method == 'ReduceThenRidgeCV':
                model = ReducedRankRegressor(regressor='Ridge')
                param_grid = [{'rank': np.linspace(1, 200, 10).astype(int),
                               'alpha': np.logspace(-4, 4, num=10)}]
                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)
            elif method == 'ReduceThenElasticNetCV':
                model = ReducedRankRegressor(regressor='ElasticNet')
                param_grid = [{'rank': np.linspace(1, 200, 10).astype(int),
                               'alpha': np.logspace(-4, 4, num=10),
                               'l1_ratio': np.linspace(0.1, 1, num=10)}]
                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)
            elif method == 'nnegElasticNet':
                model = sklinear.ElasticNet(fit_intercept=fit_intercept,
                                            positive=True, l1_ratio=0)
                # param_grid = [{'alpha': np.logspace(-4, 4, num=10),
                #                'l1_ratio': np.linspace(0.1, 1, num=10)}]
                # param_grid = [{'l1_ratio': np.linspace(0.1, 1, num=10)}]
                param_grid = [{'alpha': np.logspace(-4, 4, num=10)}]

                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)

            if tune_hyper_parameter:
                print('Tuning hyperparameter')

                # model_fit_pipeline = sklpipe.Pipeline(steps=[('regression', model)])

                # Warning about n_jobs : by default, the data is copied n_jobs * 2 many times.
                # ie. memory usage is going to be high.

                if split_group_vector is None:
                    # if no cv method provided, then htis does 5-fold cross validation .
                    grid_search = sklselection.GridSearchCV(model, param_grid, n_jobs=2,
                                                            cv=cv_split, scoring=scoring_method)

                    grid_search_results = grid_search.fit(X=X_dev, y=y_dev)
                else:
                    print('Using custom groups to do CV splits')
                    # note that GroupKFold is not random
                    # and shuffling the input does not resolve that.
                    # cv_splitter = sklselection.GroupKFold(n_splits=cv_split)
                    cv_splitter = groupKFoldRandom(groups=split_group_vector_dev, n=cv_split, seed=cv_random_seed)
                    grid_search = sklselection.GridSearchCV(model, param_grid, n_jobs=2,
                                                            cv=cv_splitter, scoring=scoring_method)
                    grid_search_results = grid_search.fit(X=X_dev, y=y_dev, groups=split_group_vector_dev)

                best_params = grid_search_results.best_params_

                # Update the model with the best hyperparameter from cv (to be used in the test set)
                if method == 'Ridge':
                    print('Applying tuned hyperparameter to model for test set evaluation.')
                    model = sklinear.Ridge(alpha=best_params['alpha'],
                                           fit_intercept=fit_intercept, solver='auto')
                elif method == 'ReducedRankRegression':
                    try:
                        model = ReducedRankRegressor(rank=best_params['rank'],
                                                     reg=best_params['reg'])
                    except:
                        print('Seting best model failed, not sure why.')
                elif method == 'ReduceThenRidgeCV':
                    print('Applying tuned hyperparameter to model for test set evaluation.')
                    model = ReducedRankRegressor(rank=best_params['rank'],
                                                 alpha=best_params['alpha'],
                                                 regressor='Ridge')
                elif method == 'ReduceThenElasticNetCV':
                    print('Applying tuned hyperparameter to model for test set evaluation.')
                    model = ReducedRankRegressor(rank=best_params['rank'],
                                                 alpha=best_params['alpha'],
                                                 l1_ratio=best_params['l1_ratio'],
                                                 regressor='ElasticNet')

                fit_results.update(best_params=best_params,
                                   cv_hyperparam_search_results=grid_search_results.cv_results_)

                # add the indices (eg. Trials) used in development and testing set
                if split_group_vector is not None:
                    fit_results.update(dev_set_groups=split_group_vector_dev,
                                       test_set_groups=split_group_vector_test)

            else:
                # TODO: need to double check; I think default behaviour is to use unifom weighting
                cross_val_raw_r2_score = sklselection.cross_val_score(model, X_dev, y_dev,
                                                                      cv=cv_split,
                                                                      scoring='r2')

                # See this page for the available score methods
                # https://scikit-learn.org/stable/modules/model_evaluation.html

                # TODO: variance weighted may need custom scorer...
                # cross_val_var_weighted_r2_score = sklselection.cross_val_score(model, X_dev, y_dev, cv=cv_split,
                #                                                       scoring='variance_weighted')

                cross_val_explained_variance = sklselection.cross_val_score(model, X_dev, y_dev, cv=cv_split,
                                                                            scoring='explained_variance')

                # add the indices (eg. Trials) used in development and testing set
                if split_group_vector is not None:
                    fit_results.update(dev_set_groups=split_group_vector_dev,
                                       test_set_groups=split_group_vector_test)

            # Get the development and test set time bins (used for realignment latter)
            if time_bins is not None:
                fit_results.update(time_bins_dev=time_bins_dev,
                                   time_bins_test=time_bins_test)

            # Fit to the entire development set (note this operation is inplace for 'model' as well)
            test_model = model.fit(X_dev, y_dev)

            # Do a final evaluation on the test set
            y_test_prediction = model.predict(X_test)

            test_raw_r2_score = sklmetrics.r2_score(y_test, y_test_prediction, multioutput='raw_values')
            test_var_weighted_r2_score = sklmetrics.r2_score(y_test, y_test_prediction, multioutput='variance_weighted')
            test_explained_variance = sklmetrics.explained_variance_score(y_test, y_test_prediction,
                                                                          multioutput='raw_values')

            # also fit things to the development set (just for the record / plotting)
            y_dev_prediction = model.predict(X_dev)

            # Look at the variance explained in the development set, just for the record
            dev_explained_variance = sklmetrics.explained_variance_score(y_dev, y_dev_prediction,
                                                                         multioutput='raw_values')

            # Fit a non-validated model to the test set to look at the explainable variance
            non_validated_model = sklinear.Ridge(alpha=0, fit_intercept=fit_intercept, solver='auto')
            non_validated_model = non_validated_model.fit(X_test, y_test)
            non_validated_model_prediction = non_validated_model.predict(X_test)
            test_set_explainable_variance = sklmetrics.explained_variance_score(y_test, non_validated_model_prediction,
                                                                                multioutput='raw_values')

            # Store everything about the model as a Bunch
            fit_results.update(test_raw_r2_score=test_raw_r2_score,
                               test_var_weighted_r2_score=test_var_weighted_r2_score,
                               test_explained_variance=test_explained_variance,
                               y_test_prediction=y_test_prediction,
                               y_test=y_test,
                               dev_model=model, test_model=test_model,
                               y_dev=y_dev,
                               y_dev_prediction=y_dev_prediction,
                               dev_explained_variance=dev_explained_variance,
                               test_set_explainable_variance=test_set_explainable_variance)

            if save_X:
                fit_results.update(X_dev=X_dev, X_test=X_test)

            if not tune_hyper_parameter:
                fit_results.update(cross_val_raw_r2_score=cross_val_raw_r2_score,
                                   cross_val_explained_variance=cross_val_explained_variance)

            if preprocessing_method is not None:
                fit_results['preprocessing_model'] = reduction_transformer

    else:
        print('Warning: no evaluation method specified')

    if save_path is None:
        return fit_results
    else:
        # TODO: save model results
        print('Need to implement saving of model results')


def fit_multiple_models(X_dict, Y, models=None,
                        fit_intercept=True,
                        evaluation_method='fit-all', cv_split=10, num_repeats=5,
                        tune_hyper_parameter=True, split_group_vector=None, preprocessing_method=None,
                        rank=10, dev_test_random_seed=None, cv_random_seed=None,
                        save_path=None, model_stats=['test_raw_r2_score', 'test_var_weighted_r2_score',
                                                     'test_explained_variance'], bin_width=None,
                        do_compute_response_function=False, event_names=None, event_start_ends=None,
                        trial_conds_df=None, time_bins=None):
    """

    Parameters
    ----------
    X_dict : (dict)
        dictionary where key is the name of the feature set
        and the value is a feature matrix
    Y
    models
    fit_intercept
    evaluation_method
    cv_split
    num_repeats
    tune_hyper_parameter
    split_group_vector
    preprocessing_method
    rank
    dev_test_random_seed
    cv_random_seed
    save_path
    model_stats
    trial_conds_df : (pandas dataframe)
        dataframe containing the stimulus condition for each trial.
    event_names : (dict)
        dictionary where key is the feature set name
    event_start_ends : (dict)
    Returns
    -------
    fit_results_dict : (dict)
        dictionary with a dictionary inside
        first layer keys correspond to the feature set being used
        second layer keys correspond to the regression model being used
        eg. fit_results_dict['audOnly']['Ridge'] gives the results for fitting
        using audio events only and training a ridge regression model.
    """

    num_neuron = np.shape(Y)[1]
    model_comparison_df_list = list()
    fit_results_dict = dict()

    for X_name, X in X_dict.items():

        fit_results_dict[X_name] = dict()

        for model in models:
            fit_results = fit_kernel_regression(X, Y, method=model,
                                                fit_intercept=fit_intercept,
                                                evaluation_method=evaluation_method, cv_split=cv_split,
                                                num_repeats=num_repeats,
                                                tune_hyper_parameter=tune_hyper_parameter,
                                                split_group_vector=split_group_vector,
                                                preprocessing_method=preprocessing_method,
                                                rank=rank, dev_test_random_seed=dev_test_random_seed,
                                                cv_random_seed=cv_random_seed,
                                                save_path=save_path, time_bins=time_bins)

            if do_compute_response_function:
                model_response_function_dict = compute_response_function(
                    model=fit_results['test_model'],
                    event_names=event_names[X_name],
                    event_start_ends=event_start_ends[X_name],
                    bin_width=bin_width, include_intercept=True,
                )
                fit_results['model_response_function_dict'] = model_response_function_dict

            model_df = pd.DataFrame.from_dict({
                'neuron': np.arange(num_neuron)
            })

            # Add model fitting results for model comparison
            for model_stat in model_stats:
                model_df[model_stat] = fit_results[model_stat]

            model_df['X_name'] = X_name
            model_df['model'] = model  # add model name

            model_comparison_df_list.append(model_df)

            if trial_conds_df is not None:
                fit_results['trial_conds_df'] = trial_conds_df

            fit_results_dict[X_name][model] = fit_results

    model_comparison_df = pd.concat(model_comparison_df_list)

    return model_comparison_df, fit_results_dict


def compute_response_function(model, event_names, event_start_ends, bin_width,
                              include_intercept=True, diag_fill_values=None,
                              custom_event_names=None, feature_column_dict=None,
                              add_intercept_to_response=False):
    """
    Compute the 'kernel' / fitted neural activity aligned to onset of event for each neuron.
    Parameters
    --------------
    model object (assume from sklearn either from sklearn.linear or sklearn.Ridge)
    event_names: (list of str)
        names of the events
    event_start_ends : (numpy ndarray)
        matrix with shape 2 x N, where N is the number of events
        first column is the start time of events, second column is the end time of events
    bin_width : (float)
        width of time bins (seconds)
    include_intercept: (bool)
        whether the fitted model includes the intercept
    custom_event_names : ()
    feature_column_dict: (dict)
        dictionary obtained from kernel_regression.make_feature_matrix
    add_intercept_to_response : (bool)
        whether to add the fitted intercept to the response function
    """

    if hasattr(model, 'coef_'):
        # Ridge Regression Model
        full_feature_matrix_size = model.coef_
        # assuming the output is multivariate (ie. multiple neurons)
        # in other words, the model coefficient matrix is expected to be numNeuron x numEventBins
        if full_feature_matrix_size.ndim == 2:
            num_parameters = np.shape(full_feature_matrix_size)[1]
        elif full_feature_matrix_size.ndim == 1:
            num_parameters = len(full_feature_matrix_size)
    elif hasattr(model, 'A'):
        # Reduce Rank based models
        num_parameters = np.shape(model.A)[1]
    else:
        print('Model lacks property that allows for computing kernels')

    if feature_column_dict is None:
        response_function_dict = dict()
    else:
        response_function_dict = collections.defaultdict(dict)

    if feature_column_dict is None:
        event_start_index = 1

    if diag_fill_values is None:
        diag_fill_values = np.full(shape=(len(event_names, )), fill_value=1)

    for n_event, event_name in enumerate(event_names):
        event_bin_start, event_bin_end = event_start_ends[n_event, :]
        # construct toeplitz (this time only for a single 'trial' / instance)
        event_bins = np.arange(event_bin_start, event_bin_end, bin_width)
        num_event_bins = len(event_bins)
        event_toeplitz = np.zeros(shape=(num_event_bins, num_event_bins))
        np.fill_diagonal(event_toeplitz, diag_fill_values[n_event])  # modified in place

        # add the intercept (first column of ones)
        intercept_column = np.full(shape=(num_event_bins, 1), fill_value=1)

        # Fill the rest of the design matrix with zeros (ie. other events)
        event_design_matrix = np.zeros(shape=(num_event_bins, num_parameters))

        if include_intercept:
            event_design_matrix[:, 0] = intercept_column.flatten()

        if feature_column_dict is None:
            event_design_matrix[:, event_start_index:event_start_index + num_event_bins] = event_toeplitz
            event_start_index = event_start_index + num_event_bins
        else:
            event_index = feature_column_dict[event_name]
            event_design_matrix[:, event_index] = event_toeplitz

        event_response = model.predict(event_design_matrix)

        if add_intercept_to_response:
            event_response = model.intercept_

        if custom_event_names is None:
            response_function_dict[event_name + '_design_matrix'] = event_design_matrix
            response_function_dict[event_name + '_response'] = event_response
            response_function_dict[event_name + '_bins'] = event_bins
        else:
            response_function_dict[custom_event_names[n_event]]['design_matrix'] = event_design_matrix
            response_function_dict[custom_event_names[n_event]]['response'] = event_response
            response_function_dict[custom_event_names[n_event]]['bins'] = event_bins

    return response_function_dict


def test_kernel_significance(model, X, y, feature_column_dict,
                             method='leave-one-out', num_cv_folds=5,
                             split_group_vector=None,
                             sig_metric=['explained-variance'],
                             cv_random_seed=None, param_dict=None,
                             check_neg_features=False):
    """
    Test the significance of each kernel in the feature matrix.
    Parameters
    ----------
    model  : scikit-learn model class
        sklearn model object with set hyperparameters (but not fitted to data yet)
        can also be a sklearn pipeline where hyperparameter tuning is done via cross validation
        within the training set.
    X : numpy ndarray
        feature matrix of shape (num_time_bin, num_neuron)
    y : (numpy ndarray)
        target matrix of shape (num_time_bin, num_neuron)
    feature_column_dict : (dict)
        dictionary where the key are the event (kernel) names
        the values are the indices of the columns containing that kernel in the feature matrix
        note that the 0-index for the index term is not included (baseline firing rate term?)
    param_dict : (dict)
        dictionary with the parameters to set to the models
        otherwise I will arbitrarily pick 'sensible' parameters.
    split_group_vector : (numpy ndarray)
        group that is used to generate cross validation / test train splits.
        most common use case is to specify the trial number of each time bin using this vector,
        so that when you do test-train and cv splits you split by trial number (instead of randomly)
        this is useful when time bins are not independent of each other (ie. there is autocorrelation)
    cv_random_seed : (int)
    sig_metric : (list of str)
        metric(s) to determine whether each kernel is significant
        eg. Steinmetz 2018 define a significant kernel as having more than 2% variacne
        explained in the test set (on average)
    Returns
    -------
    kernel_sig_results pandas dataframe
        dataframe with colummns : (1) kernel name (2) neuron (3) explained variance (4) cv number
    """

    # TODO: allow parallelising this (both cv part and event part can be parallelised)

    # check that split_group_vector has the correct number of values
    if split_group_vector is not None:
        assert len(split_group_vector) == np.shape(X)[0]

    # Apply chosen parameters to model
    # Not actually sure if this will work ....
    # Actually preferred if provided model already has the tune hyperparameters.
    if param_dict is not None:
        for param_name, param_val in param_dict.items():
            model.param_name = param_val

    df_store = list()
    num_neuron = np.shape(y)[1]

    if method == 'leave-one-out':

        for event, feature_column_index in feature_column_dict.items():

            # generate cross-validation split
            if split_group_vector is not None:
                cv_splitter = groupKFoldRandom(groups=split_group_vector,
                                               n=num_cv_folds, seed=cv_random_seed)
            else:
                print('Implement CV without groups')

            for fold_n, (train_idx, test_idx) in enumerate(cv_splitter):
                X_train = X[train_idx, :]
                X_test = X[test_idx, :]
                y_train = y[train_idx, :]
                y_test = y[test_idx, :]

                # subset the feature matrix to remove the kernel of interest
                X_train_leave_one = X_train.copy()  # separate objects!
                X_test_leave_one = X_test.copy()
                X_train_leave_one[:, feature_column_dict[event]] = 0
                X_test_leave_one[:, feature_column_dict[event]] = 0

                # if (event == 'audRightOne') & (fold_n == 3):
                #     pdb.set_trace()

                # Fit model with leave-one-out feature matrix
                try:
                    train_fit_model = model.fit(X_train_leave_one, y_train)
                except:
                    print('One kernel fitting on X_train_leave_one failed, '
                          'this is using event: %s in fold %.f.'
                          'Try making reg a small positive value.' % (event, fold_n))

                # Predict training and testing data and subtract the difference from actual training and testing
                train_prediction = train_fit_model.predict(X_train_leave_one)
                train_diff = y_train - train_prediction

                test_prediction = train_fit_model.predict(X_test_leave_one)
                test_diff = y_test - test_prediction

                # Train error on the one kernel feature matrix, and evaluate it's performance on the test set residuals
                one_kernel_X_train = X_train - X_train_leave_one

                try:
                    one_kernel_train_fit_model = model.fit(one_kernel_X_train, train_diff)
                except:
                    print('One kernel fitting failed, this is using event: %s' % event)
                # one_kernel_train_prediction = one_kernel_train_fit_model.predict(one_kernel_X_train)  # for plotting

                one_kernel_X_test = X_test - X_test_leave_one
                one_kernel_test_prediction = one_kernel_train_fit_model.predict(one_kernel_X_test)

                if 'explained-variance' in sig_metric:
                    explained_variance_score = sklmetrics.explained_variance_score(test_diff,
                                                                                   one_kernel_test_prediction,
                                                                                   multioutput='raw_values')
                    event_cv_df = pd.DataFrame.from_dict({'Explained Variance': explained_variance_score})

                event_cv_df['neuron'] = np.arange(0, num_neuron)
                event_cv_df['cv_number'] = fold_n
                event_cv_df['event'] = event

                df_store.append(event_cv_df)

        kernel_sig_results = pd.concat(df_store)

        return kernel_sig_results

    elif method == 'compare-two-models':

        # Compute full model fit
        all_cv_full_model_fit_df_list = list()

        if split_group_vector is not None:
            cv_splitter = groupKFoldRandom(groups=split_group_vector,
                                           n=num_cv_folds, seed=cv_random_seed)
        for fold_n, (train_idx, test_idx) in enumerate(cv_splitter):
            X_train = X[train_idx, :]
            X_test = X[test_idx, :]
            y_train = y[train_idx, :]
            y_test = y[test_idx, :]

            train_fit_model = model.fit(X_train, y_train)
            test_prediction = train_fit_model.predict(X_test)
            explained_variance_score = sklmetrics.explained_variance_score(y_test,
                                                                           test_prediction,
                                                                           multioutput='raw_values')
            full_model_fit_df = pd.DataFrame.from_dict({'Explained Variance': explained_variance_score})

            full_model_fit_df['neuron'] = np.arange(0, num_neuron)
            full_model_fit_df['cv_number'] = fold_n
            full_model_fit_df['event'] = 'full'

            all_cv_full_model_fit_df_list.append(full_model_fit_df)

        all_cv_full_model_fit_df = pd.concat(all_cv_full_model_fit_df_list)

        # Compute fits of model removing each events individually
        for event, feature_column_index in feature_column_dict.items():

            for fold_n, (train_idx, test_idx) in enumerate(cv_splitter):
                X_train = X[train_idx, :]
                X_test = X[test_idx, :]
                y_train = y[train_idx, :]
                y_test = y[test_idx, :]

                # subset the feature matrix to remove the kernel of interest
                X_train_leave_one = X_train.copy()  # separate objects!
                X_test_leave_one = X_test.copy()
                X_train_leave_one[:, feature_column_dict[event]] = 0
                X_test_leave_one[:, feature_column_dict[event]] = 0

                loo_fit_model = model.fit(X_train_leave_one, y_train)
                loo_test_prediction = loo_fit_model.predict(X_test_leave_one)

                if 'explained-variance' in sig_metric:
                    explained_variance_score = sklmetrics.explained_variance_score(y_test,
                                                                                   loo_test_prediction,
                                                                                   multioutput='raw_values')
                    event_cv_df = pd.DataFrame.from_dict({'Explained Variance': explained_variance_score})

                event_cv_df['neuron'] = np.arange(0, num_neuron)
                event_cv_df['cv_number'] = fold_n
                event_cv_df['event'] = event

                df_store.append(event_cv_df)

        kernel_sig_results = pd.concat(df_store)

        return all_cv_full_model_fit_df, kernel_sig_results


def get_stim_cond_fit(exp_fit_results, exp_behave_df,
                      aud_diff=np.inf, vis_diff=0.8, cell_idx=0):
    """
    Get kernel regression fit and ground truth.
    Currently only works for passive condition
    Parameters
    ----------
    exp_fit_results : (dict)
        model fitting results for a single experiment
        eg. files in '/media/timsit/Partition 1/data/interim/regression-model/passive-data/additive-model-smooth-3-20-ridge-tune
    exp_behave_df : (pandas dataframe)
        behaviour dataframe for that specific experiment
    aud_diff : (float or np.inf)
        auditory difference: np.inf (audio off), -60 (left), or 60 (right)
    vis_diff : (float)
        visual diffference: 0 (visual off) -0.8 (left) 0.8 (right)
    Returns
    -------

    """

    dev_set_trials = exp_fit_results['dev_set_groups']
    dev_prediction_matrix = exp_fit_results['y_dev_prediction']
    dev_set_data_matrix = exp_fit_results['y_dev']

    # TODO: NOTE: this may need to be converted to loc no???
    dev_set_exp_behave_df = exp_behave_df.iloc[
        pd.unique(dev_set_trials)
    ]

    dev_set_exp_behave_df = dev_set_exp_behave_df.set_index(
        pd.unique(dev_set_trials))

    dev_stim_cond_trials = dev_set_exp_behave_df.loc[
        (dev_set_exp_behave_df['audDiff'] == aud_diff) &
        (dev_set_exp_behave_df['visDiff'] == vis_diff)
        ]

    dev_stim_cond_idx = np.where(
        np.isin(dev_set_trials, dev_stim_cond_trials.index))[0]

    stim_cond_dev_set_prediction = dev_prediction_matrix[
        dev_stim_cond_idx, cell_idx]

    # TEMP HACK
    # num_time_bins = 90
    # stim_cond_dev_set_prediction = stim_cond_dev_set_prediction[
    #     0:int(num_time_bins * len(dev_stim_cond_trials))
    # ']

    stim_cond_dev_set_prediction = np.reshape(
        stim_cond_dev_set_prediction,
        newshape=(len(dev_stim_cond_trials), -1)
    )

    stim_cond_dev_set_truth = dev_set_data_matrix[
        dev_stim_cond_idx, cell_idx]

    # TEMP HACK
    # stim_cond_dev_set_truth = stim_cond_dev_set_truth[
    #     0:int(num_time_bins * len(dev_stim_cond_trials))
    # ]

    stim_cond_dev_set_truth = np.reshape(
        stim_cond_dev_set_truth,
        newshape=(len(dev_stim_cond_trials), -1)
    )

    return stim_cond_dev_set_prediction, stim_cond_dev_set_truth


def extract_regression_fit_results(model_result_folder,
                                   hyperparam_tuned=True,
                                   include_dev_var=True,
                                   include_test_var=True,
                                   include_explainable_variance=False,
                                   explainable_variance_var_name='test_set_explainable_variance',
                                   test_var_name='test_explained_variance',
                                   test_weighted_r2_name='test_var_weighted_r2_score',
                                   dev_var_name='dev_explained_variance',
                                   random_seed=None):
    """
    Extract the regerssion fit results (multiple bunch files in saved in .pkl form) to dataframes.
    Parameters
    ----------
    model_result_folder : (str)
        path to the folder containing .pkl files (bunch objects)
    hyperparam_tuned : (bool)
        whether the hyperparmaeters were tuned in the model result, if True, then hyperparameter
        related data will be extracted
    include_dev_var : (bool)

    include_test_var
    include_explainable_variance : (bool)
        whether to include the expalinable variance on the test set, which is obtained
        by fititng a model on the test set and evaluating also on the test set.
    test_var_name
    test_weighted_r2_name
    dev_var_name
    random_seed

    Returns
    -------

    """
    var_explained_dict = collections.defaultdict(list)
    r2_dict = collections.defaultdict(list)

    if include_dev_var:
        dev_var_explained_dict = collections.defaultdict(list)

    if hyperparam_tuned:
        hyperparam_dict = collections.defaultdict(list)

    if random_seed is not None:
        fname_list = glob.glob(os.path.join(model_result_folder,
                                            '*random_seed_%.f.pkl' % random_seed))
    else:
        fname_list = glob.glob(os.path.join(model_result_folder, '*.pkl'))

    for fname in fname_list:
        # open file
        with open(fname, 'rb') as handle:
            fit_result = pkl.load(handle)

        if include_test_var:
            # get the overall variance weighted R^2 scored
            test_var_weighted_r2_score = fit_result[test_weighted_r2_name]

            # get the test set variance explained per neuron
            p_explained_var_per_neuron = fit_result[test_var_name] * 100

        # get the dev set variance explained per neuron
        if include_dev_var:
            dev_p_explained_var_per_neuron = fit_result[dev_var_name] * 100

        exp_num = fit_result['exp']
        # exp_num = int(os.path.basename(fname).split('.')[0])

        num_neuron = len(p_explained_var_per_neuron)
        neuron_idx = np.arange(num_neuron)

        var_explained_dict['neuronIdx'].extend(neuron_idx)
        var_explained_dict['pVarExplained'].extend(p_explained_var_per_neuron)
        var_explained_dict['expRef'].extend(np.repeat(exp_num, num_neuron))

        if include_dev_var:
            dev_var_explained_dict['neuronIdx'].extend(neuron_idx)
            dev_var_explained_dict['pVarExplained'].extend(dev_p_explained_var_per_neuron)
            dev_var_explained_dict['expRef'].extend(np.repeat(exp_num, num_neuron))

        if include_explainable_variance:
            var_explained_dict['explainableVariance'].extend(fit_result[explainable_variance_var_name])

        if hyperparam_tuned:
            hyperparm_df = pd.DataFrame.from_dict(
                fit_result['cv_hyperparam_search_results'])

            best_param_df = hyperparm_df.loc[
                hyperparm_df['rank_test_score'] == 1
                ]

            hyperparam_dict['meanCVscore'].append(
                best_param_df['mean_test_score'].values[0])
            hyperparam_dict['expRef'].append(exp_num)

        r2_dict['expRef'].append(exp_num)
        r2_dict['r2score'].append(test_var_weighted_r2_score)

    r2_df = pd.DataFrame.from_dict(r2_dict)
    var_explained_df = pd.DataFrame.from_dict(var_explained_dict)

    if hyperparam_tuned:
        hyperparam_cv_df = pd.DataFrame.from_dict(
            hyperparam_dict
        )
    else:
        hyperparam_cv_df = None

    if include_dev_var:
        dev_var_explained_df = pd.DataFrame.from_dict(
            dev_var_explained_dict)
    else:
        dev_var_explained_df = None

    return r2_df, var_explained_df, hyperparam_cv_df, dev_var_explained_df


def get_all_hyperparam_df(model_folder):
    fname_list = glob.glob(os.path.join(model_folder, '*.pkl'))
    all_exp_hyperparam_df_list = list()

    for fname in fname_list:
        # open file
        with open(fname, 'rb') as handle:
            fit_result = pkl.load(handle)

        exp_num = int(os.path.basename(fname).split('.')[0])

        hyperparam_df = pd.DataFrame.from_dict(
            fit_result['cv_hyperparam_search_results'])

        hyperparam_df['expRef'] = exp_num

        all_exp_hyperparam_df_list.append(hyperparam_df)

    all_exp_hyperparam_df = pd.concat(all_exp_hyperparam_df_list)

    return all_exp_hyperparam_df


def combine_variance_explained_and_ANOVA_result(combined_model_var_explained_df,
                                                anova_result_df):
    """
    Combines the variance explained of each neuron (either in the development
    or in the testing set) with the ANOVA p-values

    Parameters
    ----------
    combined_model_var_explained_df (pandas dataframe)
    anova_result_df (pandas dataframe)

    Returns
    -------
    all_exp_anova_and_kernel_df (pandas dataframe)
    """

    all_exp_anova_and_kernel_df_list = list()
    for exp in np.unique(anova_result_df['expRef']):

        exp_anova_result_df = anova_result_df.loc[
            anova_result_df['expRef'] == exp
            ]

        exp_combined_model_var_explained_df = combined_model_var_explained_df.loc[
            combined_model_var_explained_df['expRef'] == exp
            ]

        if len(exp_anova_result_df) == len(exp_combined_model_var_explained_df):

            exp_anova_and_regression_result_df = exp_anova_result_df.set_index(
                ['expRef', 'Cell']
            ).join(
                exp_combined_model_var_explained_df.set_index(['expRef', 'Cell'])
            )

            all_exp_anova_and_kernel_df_list.append(exp_anova_and_regression_result_df)


        else:
            # This happens because of ANOVA NaN (no firing rate)
            # Try to drop na to resolve the problem
            exp_anova_result_df = exp_anova_result_df.dropna()

            if len(exp_anova_result_df) == len(exp_combined_model_var_explained_df):
                exp_anova_and_regression_result_df = exp_anova_result_df.set_index(
                    ['expRef', 'Cell']
                ).join(
                    exp_combined_model_var_explained_df.set_index(['expRef', 'Cell'])
                )

                all_exp_anova_and_kernel_df_list.append(exp_anova_and_regression_result_df)
            else:

                print('Exp %.f is excluded, neuron numbers not equal' % exp)

    all_exp_anova_and_kernel_df = pd.concat(all_exp_anova_and_kernel_df_list)

    return all_exp_anova_and_kernel_df


def single_test_kernel_sig(X, Y, model, feature_column_dict, event, trial_vector,
                           num_cv_folds=10, cv_random_seed=1):
    """
    Test for the significance of a particular kernel by comparing the full model with the kernel and a model
    with the particular kernel of interest removed
    Parameters
    ----------
    X : numpy
    Y
    model
    feature_column_dict
    event
    trial_vector
    num_cv_folds
    cv_random_seed

    Returns
    -------

    """
    # model = sklinear.Ridge(alpha=1, fit_intercept=True, solver='auto')
    # model = ReducedRankRegressor(
    #     regressor='Ridge', alpha=1)
    # model = kernel_regression.ReducedRankRegressor(regressor='ElasticNet')
    # model = kernel_regression.ReducedRankRegressor(rank=50, reg=1.0)

    feature_column_index = feature_column_dict[event]
    cv_splitter = groupKFoldRandom(
        groups=trial_vector,
        n=num_cv_folds, seed=cv_random_seed)

    train_idx = cv_splitter[0][0]
    test_idx = cv_splitter[0][1]

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = Y[train_idx, :]
    y_test = Y[test_idx, :]

    # subset the feature matrix to remove the kernel of interest
    X_train_leave_one = X_train.copy()  # separate objects!
    X_test_leave_one = X_test.copy()

    X_train_leave_one[:, feature_column_dict[event]] = 0
    X_test_leave_one[:, feature_column_dict[event]] = 0

    # Fit model with leave-one-out feature matrix
    train_fit_model = model.fit(X_train_leave_one, y_train)

    # Predict training and testing data and subtract the difference from actual training and testing
    train_prediction = train_fit_model.predict(X_train_leave_one)
    train_diff = y_train - train_prediction

    test_prediction = train_fit_model.predict(X_test_leave_one)
    test_diff = y_test - test_prediction

    explained_variance_per_neuron_train = sklmetrics.explained_variance_score(
        y_true=y_train, y_pred=train_prediction, multioutput='raw_values')

    # Train error on the one kernel feature matrix, and evaluate it's performance on the test set residuals
    one_kernel_X_train = X_train - X_train_leave_one
    one_kernel_train_fit_model = model.fit(one_kernel_X_train, train_diff)

    # for illustration purposes (not needed for computation)
    one_kernel_train_prediction = one_kernel_train_fit_model.predict(one_kernel_X_train)

    one_kernel_X_test = X_test - X_test_leave_one
    one_kernel_test_prediction = one_kernel_train_fit_model.predict(one_kernel_X_test)

    explained_variance_score = sklmetrics.explained_variance_score(test_diff,
                                                                   one_kernel_test_prediction,
                                                                   multioutput='raw_values')

    return train_idx, test_idx, train_prediction, train_diff, one_kernel_train_prediction, test_prediction, one_kernel_test_prediction, \
           y_test, test_diff, explained_variance_score


def get_fitted_and_residuals(exp_behave_df, trial_vector, train_idx, test_idx,
                             y_train, train_prediction, train_diff,
                             one_kernel_train_prediction, test_prediction,
                             one_kernel_test_prediction, y_test, test_diff,
                             cell_idx, aud_diff, vis_diff):
    """
    Get the fitted kernel using leave-one-out features and the residuals.

    Parameters
    ----------
    exp_behave_df
    trial_vector
    train_idx
    test_idx
    y_train
    train_prediction
    train_diff
    one_kernel_train_prediction
    test_prediction
    one_kernel_test_prediction
    cell_idx
    aud_diff
    vis_diff

    Returns
    -------

    """

    train_set_trials = trial_vector[train_idx]
    test_set_trials = trial_vector[test_idx]

    test_set_exp_behave_df = exp_behave_df.loc[pd.unique(test_set_trials)]
    train_set_exp_behave_df = exp_behave_df.loc[pd.unique(train_set_trials)]

    train_stim_cond_trials = train_set_exp_behave_df.loc[
        (train_set_exp_behave_df['audDiff'].isin(aud_diff)) &
        (train_set_exp_behave_df['visDiff'].isin(vis_diff))
        ]

    test_stim_cond_trials = test_set_exp_behave_df.loc[
        (test_set_exp_behave_df['audDiff'].isin(aud_diff)) &
        (test_set_exp_behave_df['visDiff'].isin(vis_diff))
        ]

    train_stim_cond_idx = np.where(
        np.isin(train_set_trials, train_stim_cond_trials.index))[0]

    test_stim_cond_idx = np.where(
        np.isin(test_set_trials, test_stim_cond_trials.index))[0]

    # Get Y predicted and actual Y for specific stimulus conditions

    stim_cond_train_set_prediction = train_prediction[
        train_stim_cond_idx, cell_idx]

    stim_cond_train_set_prediction = np.reshape(
        stim_cond_train_set_prediction,
        newshape=(len(train_stim_cond_trials), -1)
    )

    stim_cond_train_set_truth = y_train[
        train_stim_cond_idx, cell_idx
    ]

    stim_cond_train_set_truth = np.reshape(
        stim_cond_train_set_truth,
        newshape=(len(train_stim_cond_trials), -1)
    )

    # get the residual used for training
    stim_cond_train_residual = train_diff[
        train_stim_cond_idx, cell_idx
    ]

    stim_cond_train_residual = np.reshape(
        stim_cond_train_residual,
        newshape=(len(train_stim_cond_trials), -1)
    )

    # get fitted residual using one kernel
    stim_cond_train_residual_prediction = one_kernel_train_prediction[
        train_stim_cond_idx, cell_idx
    ]

    stim_cond_train_residual_prediction = np.reshape(
        stim_cond_train_residual_prediction,
        newshape=(len(train_stim_cond_trials), -1)
    )

    # Test set ground truth and leave-one-out fit

    stim_cond_test_set_truth = y_test[
        test_stim_cond_idx, cell_idx
    ]

    stim_cond_test_set_truth = np.reshape(
        stim_cond_test_set_truth,
        newshape=(len(test_stim_cond_trials), -1)
    )

    stim_cond_test_prediction = test_prediction[
        test_stim_cond_idx, cell_idx
    ]

    stim_cond_test_prediction = np.reshape(
        stim_cond_test_prediction,
        newshape=(len(test_stim_cond_trials), -1)
    )

    # Residual in the test set
    stim_cond_test_residual = test_diff[
        test_stim_cond_idx, cell_idx
    ]

    stim_cond_test_residual = np.reshape(
        stim_cond_test_residual,
        newshape=(len(test_stim_cond_trials), -1)
    )

    # Prediction of the one kernel in the test set
    stim_cond_test_residual_prediction = one_kernel_test_prediction[
        test_stim_cond_idx, cell_idx
    ]

    stim_cond_test_residual_prediction = np.reshape(
        stim_cond_test_residual_prediction,
        newshape=(len(test_stim_cond_trials), -1)
    )

    return stim_cond_train_set_truth, stim_cond_train_set_prediction, \
           stim_cond_train_residual, stim_cond_train_residual_prediction, \
           stim_cond_test_set_truth, stim_cond_test_prediction, \
           stim_cond_test_residual, stim_cond_test_residual_prediction


def get_neuron_psth(behave_df, spike_ds, cell_idx=None, event_name='visLeftAudOff',
                    event_bin_start=-0.05, event_bin_end=0.4, time_coord_name='bin_start'):
    """
    Get the PSTH for a particular event. Basically does alignment.
    Parameters
    ----------
    behave_df
    spike_ds
    cell_idx
    event_name
    event_bin_start
    event_bin_end
    time_coord_name

    Returns
    -------

    """

    event_time = get_event_times(
        behave_df=behave_df, event_name=event_name
    )

    window_start = event_time + event_bin_start
    window_end = event_time + event_bin_end

    event_bins = list(zip(window_start,
                          window_end))

    event_interval_bins = pd.IntervalIndex.from_tuples(event_bins, closed='both')
    aligned_xarray_tuple = spike_ds.groupby_bins(time_coord_name, event_interval_bins)
    aligned_xarray_list = [i[1] for i in list(aligned_xarray_tuple)]

    new_aligned_xarray_list = list()

    for n_xarray, aligned_xarray in enumerate(aligned_xarray_list):
        peri_event_time = aligned_xarray[time_coord_name].values - event_time[n_xarray]
        overall_time = aligned_xarray[time_coord_name].values
        aligned_xarray = aligned_xarray.assign({'BinTime': (time_coord_name, overall_time)})
        aligned_xarray = aligned_xarray.assign({'PeriEventTime': (time_coord_name, peri_event_time)})
        aligned_xarray = aligned_xarray.assign_coords({time_coord_name: np.arange(len(overall_time))})
        new_aligned_xarray_list.append(aligned_xarray)

    # Concatenate alignment ds
    aligned_ds = xr.concat(new_aligned_xarray_list, dim='Trial')

    if cell_idx is not None:
        aligned_ds = aligned_ds.isel(Cell=cell_idx)

    return aligned_ds


def get_neuron_weights(response_function_dict, event_names):
    """
    Get the summed prediction / coefficients for each event for a neuron.
    This is used as a proxy for neuron selectivity. (if a neuron has a large response to visual right,
    then the visual right coeffcients should be large)
    Parameters
    ----------
    response_function_dict
    event_names

    Returns
    -------

    """

    num_neuron = np.shape(response_function_dict['audDiff_response'])[1]

    neuron_weight_df = pd.DataFrame.from_dict({
        'neuron': np.arange(num_neuron)
    })

    for event_name in event_names:
        response_matrix = response_function_dict[event_name + '_response']
        neuron_weight_df[event_name + '_windowSum'] = np.sum(np.abs(response_matrix), axis=0)

    return neuron_weight_df


def extract_aligned_predictions(fit_result, trial_conds_df, trial_conds_to_plot, trial_cond_names,
                                cell_idx=0,
                                actual_data_field='y_test', prediction_field='y_test_prediction',
                                groups_to_plot_field='test_set_groups', time_bins_field='time_bins_test',
                                max_num_time_bin=90):
    # pdb.set_trace()
    y_to_plot = fit_result[actual_data_field]

    y_prediction_to_plot = fit_result[prediction_field]

    if groups_to_plot_field is not None:
        groups_to_plot = fit_result[groups_to_plot_field]
    else:
        groups_to_plot = np.arange(len(trial_conds_df))

    time_bins_test = fit_result[time_bins_field]

    trial_conds_test_df = trial_conds_df.loc[
        trial_conds_df['trial_idx'].isin(groups_to_plot)
    ]

    model_cond_prediction = dict()
    actual_cond = dict()

    for n_trial_cond, trial_cond in enumerate(trial_conds_to_plot):

        subset_df = trial_conds_test_df.copy()

        for cond_name, cond_target_value in trial_cond.items():
            subset_df = subset_df.loc[
                subset_df[cond_name] == cond_target_value
                ]

        target_trial_idx = subset_df['trial_idx'].values
        num_trial_w_stim_cond = len(subset_df)

        # pdb.set_trace()
        # num_time_bin = len(np.where(groups_to_plot == groups_to_plot[0])[0])
        trial_cond_matrix = np.zeros((len(target_trial_idx), max_num_time_bin))
        y_prediction_matrix = np.zeros((len(target_trial_idx), max_num_time_bin))

        for n_trial, trial in enumerate(target_trial_idx):
            trial_time_idx_to_get = np.where(groups_to_plot == trial)[0]
            stimOnTime = subset_df.iloc[n_trial]['stimOnTime']
            # pdb.set_trace()
            subset_time_window = [stimOnTime - 0.05, stimOnTime + 0.4]
            subset_time_idx = np.where(
                (time_bins_test >= subset_time_window[0]) &
                (time_bins_test <= subset_time_window[1])
            )[0]
            # pdb.set_trace()
            y_prediction_matrix[n_trial, :] = y_prediction_to_plot[subset_time_idx, cell_idx]
            trial_cond_matrix[n_trial, :] = y_to_plot[subset_time_idx, cell_idx]

        trial_cond_mean = np.mean(trial_cond_matrix, axis=0)

        rows_to_include = np.where(np.sum(y_prediction_matrix, axis=1) != 0)[0]
        trial_cond_prediction_mean = np.mean(y_prediction_matrix[rows_to_include, :], axis=0)

        model_cond_prediction[trial_cond_names[n_trial_cond]] = trial_cond_prediction_mean
        actual_cond[trial_cond_names[n_trial_cond]] = trial_cond_mean

    return model_cond_prediction, actual_cond


def main():
    """
    Main script for running regression

    Settings
    --------


    """
    # TODO: simplify below to take in just the exp folder? (need to know how data is stored)
    DATA_FOLDER_PATH = '/media/timsit/Partition 1/data/interim/passive-m2-good-reliable-movement/subset/'
    EPHYS_BEHAVE_FILE = 'ephys_behaviour_df.pkl'
    EPHYS_CELL_FILE = 'neuron_df.pkl'
    EPHYS_SPIKE_FILE = 'spike_df.pkl'
    MODEL_TYPE = 'additive'
    fit_method = 'Ridge'
    evaluation_method = 'train-cv-test'
    target_cell_loc = ['MOs']
    bin_width = 5 / 1000
    cv_split = 5
    cv_split_repeats = 5
    smooth_sigma = 3
    smooth_window_width = 20
    smooth_spikes = True
    tune_hyper_parameter = True
    SAVE_FOLDER = '/media/timsit/Partition 1/data/interim/regression-model/passive-data/additive-model-smooth-3-20-ridge-tuned-0p8-contrast-only/'

    # because subject 1 does not have all conditions
    custom_subject_list = [2, 3, 4, 5]
    # somehow exp 32 is not working

    ephys_behave_df = pd.read_pickle(os.path.join(DATA_FOLDER_PATH, EPHYS_BEHAVE_FILE))
    neuron_df = pd.read_pickle(os.path.join(DATA_FOLDER_PATH, EPHYS_CELL_FILE))
    spike_df = pd.read_pickle(os.path.join(DATA_FOLDER_PATH, EPHYS_SPIKE_FILE))

    if MODEL_TYPE == 'additive':
        # only using high visual contrast
        event_names = ['visLeft0p8One', 'visRight0p8One',
                       'audLeftOne', 'audRightOne']

        event_start_ends = np.array(
            [[-0.05, 0.4],
             [-0.05, 0.4],
             [-0.05, 0.4],
             [-0.05, 0.4]]
        )

    elif MODEL_TYPE == 'interactive':

        event_names = ['visLeft0p8AudRightOne', 'visLeft0p8AudLeftOne',
                       'visRight0p8AudRightOne', 'visRight0p8AudLeftOne',
                       'visLeft0p8AudOff', 'visRight0p8AudOff',
                       'visOffAudLeft', 'visOffAudRight']

        event_start_ends = np.array(
            [
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
                [-0.05, 0.4],
            ]
        )
    else:
        print('WARNING: no valid model type specified')

    # subject_list = np.unique(ephys_behave_df['subjectRef'])
    for subject in custom_subject_list:

        subject_df = ephys_behave_df.loc[
            ephys_behave_df['subjectRef'] == subject
        ]

        for exp in np.unique(subject_df['expRef']):

            exp_behave_df = subject_df.loc[
                subject_df['expRef'] == exp
            ]

            neuron_id = neuron_df.loc[
                (neuron_df['expRef'] == exp) &
                (neuron_df['cellLoc'].isin(target_cell_loc))]['cellId'].tolist()

            target_spike_df = spike_df.loc[spike_df['cellId'].isin(neuron_id)]


            if len(target_spike_df) == 0:
                print('Experiment %.f has no MOs neurons, skipping...' % exp)
                continue

            # make spike matrix
            target_spike_ds = spike_df_to_xarray(spike_df=target_spike_df,
                                                        time_bin_width=bin_width)

            # Smooth spikes
            if smooth_spikes:
                target_spike_ds['spike_rate'] = (['cell', 'bin_start'], anaspikes.smooth_spikes(
                    target_spike_ds['spike_rate'],
                    method='half_gaussian',
                    sigma=smooth_sigma, window_width=smooth_window_width,
                    custom_window=None))

            spike_bins = target_spike_ds.attrs['time_bins']
            bin_width = target_spike_ds.attrs['time_bin_length']
            Y = target_spike_ds.spike_rate.values.T

            trial_vector = make_trial_vec(behave_df=exp_behave_df, spike_ds=target_spike_ds)
            # Make feature matrix
            X, feature_column_dict = make_feature_matrix(
                exp_behave_df, event_names=event_names,
                event_start_ends=event_start_ends,
                spike_bins=spike_bins, bin_width=bin_width,
                custom_feature_names=None,
                feat_mat_type='fill-diagonal-toeplitz')

            # Remove all-zero rows (not counting the intercept)
            X, Y, trial_vector = remove_all_zero_rows(X=X, Y=Y,
                                                       group_vector=trial_vector,
                                                       intercept_included=True)

            # Double check that there is no NaN in the trial vector
            # if so, likely means the there is a bug / time bins are too long
            assert not np.isnan(np.sum(trial_vector))

            # Fit model

            fit_results = fit_kernel_regression(
                          X, Y=Y, method=fit_method,
                          fit_intercept=True,
                          evaluation_method=evaluation_method,
                          cv_split=cv_split, num_repeats=cv_split_repeats,
                          tune_hyper_parameter=tune_hyper_parameter,
                          split_group_vector=trial_vector,
                          preprocessing_method=None,
                          rank=10, dev_test_random_seed=None, cv_random_seed=None,
                          save_path=None)

            # save results
            with open(os.path.join(SAVE_FOLDER, '%.f.pkl' % exp), 'wb') as handle:
                pkl.dump(fit_results, handle)


if __name__ == '__main__':
    main()