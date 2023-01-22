from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import dummy
import sklearn as skl
import sklearn.model_selection as sklselection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
import sklearn.linear_model as sklinear
from sklearn.utils import resample
import sklearn.metrics as sklmetrics
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import random

import src.data.xarray_util as xr_util

# some sklearn models
from sklearn.svm import SVC
from src.models.kernel_regression import groupKFoldRandom

import scipy.stats as sstats

# general util
import itertools
import bunch
import pickle as pkl
import os
import glob
import pdb   # debugging


def predict_decision(X, y, clf, control_clf='default', get_importance=True, loading_bar=False,
                     n_cv_splits=5, n_repeats=5, chunk_vector=None, cv_random_seed=None,
                     feature_importance_type='list', feature_names=None, extra_feature_fields=None,
                     include_feature_importance=True, extra_X=None, extra_y=None, match_y_extra_dist=False,
                     cv_split_method='stratifiedKFold', tune_hyperparam=False,
                     n_inner_loop_cv_splits=5, n_inner_loop_repeats=2,
                     hyperparam_tune_method='nested_cv_w_loop', param_grid=[{'C':  np.logspace(-5, 2, 11)}],
                     param_search_scoring_method=None, include_confidence_score=False,
                     aud_cond=None, vis_cond=None, classifier_object_name='svc',
                     trial_cond=None):
    """
    Evaluates classifier performance using repeated stratified k-fold cross-validation.
    NOTE: This function is going to be supercedeed by predict_decision_multi_clf

    Parameters
    -----------
    X : numpy ndarray
        feature matrix of shape (num_samples, num_features)
    y : numpy ndarray
        label vector of shape (num_samples, )
    clf : sklearn classifier object
        classifier to use
    control_clf : str, sklearn classifier object, NoneType
        if (1) 'default' - runs a dummy cer based on class distribution in training set
           (2) None      - does not run a control classfier
           (3) sklearn classifier object, any classifier you want to compare with
    get_importance : bool
        if True, attempts to get the feature importance (eg. coefficients) of the model
            in linear sklearn models, this will work, and should return coefs with shape
                (num_class, num_features)
    chunk_vector : (None or numpy ndarray)
        a vector that controls the chunking in cross validation splits, such that samples within the same
        chunk are not split: they will either all be in the training set, or all in the validation set.
    loading_bar : int
        whether to include a loading bat for classification
    n_cv_splits : int
        number of cross-validation (cv) splits to perform
    n_repeats : int
        number of repeats of the cross-validation process to perform
        (each time with a different set of random CV partitions)
    cv_random_seed : int
        random seed for cross-validation partitions
    extra_X : numpy ndarray
        feature matrix of shape (num_samples, num_features) that you want use your trained model to make predictions about on top
        of the testing set (eg. if you want to see if a model trained in a certain dataset will generalise
        to another dataset without training on that dataset)
    extra_y : numpy ndarray
        label vector of shape (num_samples, ) corresponding to the feature matrix in extra_X
    classifier_object_name : str
        name of the classifier object when the clf object provided is a Pipeline object.
        this is needed to get the coefficients of the classifier object.
    extra_feature_fields : dict
        fields that you want to add to your features, eg. location of each neuron
        these will be accsesed via the dictinoary: key will be the field name, and
        the items will be the data associated with each feature.
    trial_cond : dict
        fields associated with each trial / sample (row of your feature matrix)
        this is a generalisiation of aud_cond and vis_cond, and allows for arbitrary data
        associated with each trial (eg. reaction time)
    aud_cond : int
        auditory condition for each trial
    vis_cond : int
        visual condition for each trial
    hyperparam_tune_method : str
        how to
    Returns
    --------

    # TODO: add option to subsample y_extra based on the class distribution in y_test
    """

    if control_clf is not None:
        if control_clf == 'default':
            dummy_clf = dummy.DummyClassifier(strategy='stratified')
        elif control_clf == 'most_frequent':
            dummy_clf = dummy.DummyClassifier(strategy='most_frequent')
        else:
            dummy_clf = control_clf

    # sklearn object to train-test splits
    if chunk_vector is None:
        if cv_split_method == 'stratifiedKFold':
            cv_splitter = RepeatedStratifiedKFold(n_splits=n_cv_splits, n_repeats=n_repeats)
        elif cv_split_method == 'repeatedKFold':
            cv_splitter = RepeatedKFold(n_splits=n_cv_splits, n_repeats=n_repeats)
    else:
        # TODO: add the random seed option.
        cv_splitter = groupKFoldRandom(groups=chunk_vector, n=n_cv_splits)

    accuracy_score_list = list()
    hyperparam_search_score_list = list()
    extra_accuracy_score_list = list()
    dummy_accuracy_score_list = list()
    feature_importance_list = list()

    # Include confidence metric and the audio-visual condition to be used later when looking at the metric
    # for different stimulus conditions.
    if include_confidence_score:
        confidence_score_df_list = list()

    if tune_hyperparam:

        if hyperparam_tune_method == 'nested_cv':
            print('WARNING: implementation incomplete')
            # nested cv using sklearn built in methods
            outer_cv = cv_splitter
            inner_cv = RepeatedKFold(n_splits=n_inner_loop_cv_splits, n_repeats=n_inner_loop_repeats)

            clf = sklselection.GridSearchCV(estimator=clf, param_grid=param_grid,
                                            cv=inner_cv)
            nested_cv_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
            accuracy_score_list = nested_cv_score

        elif hyperparam_tune_method == 'nested_cv_w_loop':
            # Note here that we use a nested cross validation strategy
            # See for example: https://chrisalbon.com/machine_learning/model_evaluation/nested_cross_validation/
            # But there will be high variability of results with small datasets
            # tune hyperparameter using cross validation, then do a final test on the test set

            # Outer-loop
            for n_split, (dev_index, test_index) in enumerate(cv_splitter.split(X, y)):
                X_dev, X_test = X[dev_index], X[test_index]
                y_dev, y_test = y[dev_index], y[test_index]

                # Inner loop: do cross-validation on development set to find best hyperparameter
                if chunk_vector is None:
                    # if no cv method provided, then this does 5-fold cross validation.
                    inner_loop_cv_splitter = sklselection.RepeatedStratifiedKFold(
                        n_splits=n_inner_loop_cv_splits, n_repeats=n_inner_loop_repeats)

                    grid_search = sklselection.GridSearchCV(clf, param_grid, n_jobs=-1,
                                                            cv=inner_loop_cv_splitter,
                                                            scoring=param_search_scoring_method,
                                                            refit=True, iid=True)

                    grid_search_results = grid_search.fit(X=X_dev, y=y_dev)

                    # evaluate best hyperparameter on the test set
                    best_model = grid_search_results.best_estimator_
                    accuracy_score = best_model.score(X_test, y_test)
                    accuracy_score_list.append(accuracy_score)

                    grid_search_result_df = pd.DataFrame(grid_search_results.cv_results_)
                    grid_search_result_df['n_split'] = n_split

                    hyperparam_search_score_list.append(grid_search_result_df)

                    if include_confidence_score:
                        test_confidence_score = best_model.decision_function(X_test)
                        dev_confidence_score = best_model.decision_function(X_dev)

                        if aud_cond is not None and vis_cond is not None:
                            print('Aud cond and vis cond arugment soon to be generalised'
                                  'to trial_cond, use that instead.')
                            dev_confidence_df = pd.DataFrame({
                                'confidence_score': dev_confidence_score,
                                'aud_cond': aud_cond[dev_index],
                                'vis_cond': vis_cond[dev_index],
                            })
                        else:
                            dev_confidence_df = pd.DataFrame({
                                'confidence_score': dev_confidence_score,
                            })

                        dev_confidence_df['dataset'] = 'dev'

                        if aud_cond is not None and vis_cond is not None:
                            test_confidence_df = pd.DataFrame({
                                'confidence_score': test_confidence_score,
                                'aud_cond': aud_cond[test_index],
                                'vis_cond': vis_cond[test_index]
                            })
                        else:
                            test_confidence_df = pd.DataFrame({
                                'confidence_score': test_confidence_score,
                            })

                        test_confidence_df['dataset'] = 'test'

                        if trial_cond is not None:
                            for field_name, field_data in trial_cond.items():
                                dev_confidence_df[field_name] = field_data[dev_index]
                                test_confidence_df[field_name] = field_data[test_index]

                        confidence_score_df = pd.concat([dev_confidence_df, test_confidence_df])

                        confidence_score_df['n_split'] = n_split

                        confidence_score_df_list.append(confidence_score_df)

                # Run a control classifier
                if control_clf is not None:
                    dummy_accuracy_score = dummy_clf.fit(X_dev, y_dev).score(X_test, y_test)
                    dummy_accuracy_score_list.append(dummy_accuracy_score)

                # Get the feature importance of the best model
                if get_importance is True:
                    if type(best_model) is skl.pipeline.Pipeline:
                        feature_importance = best_model[classifier_object_name].coef_
                    else:
                        feature_importance = best_model.coef_
                    if feature_importance_type == 'list':
                        feature_importance_list.append(feature_importance)
                    elif feature_importance_type == 'df':
                        if feature_names is None:
                            feature_names = np.arange(0, np.shape(feature_importance)[1])

                        feature_importance_df = pd.DataFrame.from_dict({'feature': feature_names,
                                                                        'weight': feature_importance[0, :],
                                                                        'n_split': np.repeat(n_split,
                                                                                             len(feature_names)),
                                                                        }
                                                                       )
                        # add extra information about each feature (eg. cell location of each neuron)
                        if extra_feature_fields is not None:
                            for feature_field_name, feature_field_value in extra_feature_fields.items():
                                feature_importance_df[feature_field_name] = feature_field_value

                        feature_importance_list.append(feature_importance_df)

        elif hyperparam_tune_method == 'hold_out':

            print('Implement hold out hyperparameter tuning.')

        else:
            print('No valid method selected.')

    else:

        for n_split, (train_index, test_index) in enumerate(cv_splitter.split(X, y)):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # TODO: need to do fit_trasnform if using pipeline (not sure why, got the error
            # 'StandardScalar has no attribute mean_'
            fitted_model = clf.fit(X_train, y_train)
            accuracy_score = fitted_model.score(X_test, y_test)
            accuracy_score_list.append(accuracy_score)

            if extra_X is not None and extra_y is not None:
                if match_y_extra_dist:
                    extra_y, extra_X = stratify_from_target_y(y_target=y_test,
                                                              y=extra_y, X=extra_X)
                extra_accuracy_score = fitted_model.score(extra_X, extra_y)
                extra_accuracy_score_list.append(extra_accuracy_score)

            if get_importance is True:
                if type(clf) is skl.pipeline.Pipeline:
                    feature_importance = clf[classifier_object_name].coef_
                elif 'coef_' in clf.__dict__.keys():
                    feature_importance = clf.coef_[0, :]
                else:
                    feature_importance = np.zeros((np.shape(X_train)[1], ))

                if feature_importance_type == 'list':
                    feature_importance_list.append(feature_importance)
                elif feature_importance_type == 'df':
                    if feature_names is None:
                        if 'coef_' in clf.__dict__.keys():
                            feature_names = np.arange(0, np.shape(clf.coef_)[1])
                        else:
                            feature_names = np.arange(0, np.shape(X_train)[1])

                    feature_importance_df = pd.DataFrame.from_dict({'feature': feature_names,
                                                                    #  'weight': clf.coef_[0, :],
                                                                    'weight': feature_importance,
                                                                     'n_split': np.repeat(n_split, len(feature_names)),
                                                                    }
                                                                   )

                    # add extra information about each feature (eg. cell location of each neuron)
                    if extra_feature_fields is not None:
                        for feature_field_name, feature_field_value in extra_feature_fields.items():
                            feature_importance_df[feature_field_name] = feature_field_value

                    feature_importance_list.append(feature_importance_df)

            if control_clf is not None:
                dummy_accuracy_score = dummy_clf.fit(X_train, y_train).score(X_test, y_test)
                dummy_accuracy_score_list.append(dummy_accuracy_score)

    if control_clf is None:
        dummy_accuracy_score_list.append(np.repeat(np.nan, len(accuracy_score_list)))

    if get_importance:
        if feature_importance_type == 'list':
            feature_importance_output = feature_importance_list
        elif feature_importance_type == 'df':
            feature_importance_output = pd.concat(feature_importance_list)
    else:
        feature_importance_output = None

    if len(extra_accuracy_score_list) > 0:
        accuracy_score_output = {'original_condition': accuracy_score_list,
                                 'extra_condition': extra_accuracy_score_list}
    else:
        accuracy_score_output = accuracy_score_list

    if tune_hyperparam:
        hyperparam_tune_output = pd.concat(hyperparam_search_score_list)
    else:
        hyperparam_tune_output = None

    if include_confidence_score:
        confidence_score_output = pd.concat(confidence_score_df_list)
    else:
        confidence_score_output = None

    return accuracy_score_output, dummy_accuracy_score_list, feature_importance_output,\
           hyperparam_tune_output, confidence_score_output


def run_window_classification(activity_dataset, labels, clf, num_bins=50, window_width=20, num_window=10,
                              window_width_units='bins', custom_window_locs=None,
                              even_steps=True, run_parallel=False, n_cv_splits=5, n_repeats=5,
                              cv_split_method='stratifiedKFold',
                              random_shuffle_labels=False, window_type='center',
                              feature_importance_type='list', include_cell_loc=False,
                              include_peri_event_time=False,
                              print_loading_bar=False, get_importance=True,
                              control_clf='most_frequent', activity_name='firing_rate',
                              extra_condition_activity_dataset=None, extra_labels=None,
                              include_baseline_accuracy=False, tune_hyperparam=False,
                              n_inner_loop_cv_splits=5, n_inner_loop_repeats=2,
                              hyperparam_tune_method='nested_cv_w_loop', param_grid=[{'C': np.logspace(-5, 2, 11)}],
                              param_search_scoring_method=None, scale_features=False, include_confidence_score=False,
                              peri_event_time_name='PeriEventTime', trial_cond=None,
                              ):
    """
    Generates a feature matrix from each time window, and run classifiers to predict the event identity.
    Parameters
    ----------
    activity_dataset      : xarray dataset
        object containing the feature matrix for classification
    labels                : (numpy ndarray)
        (num_trial, ) array with int labels (eg. left vs. right choice) to classify
    num_bins              : (int)
        number of time bins in the entire aligned time window
    window_width          : (int)
        width (in time bins) of the window to average activity over
    num_window            : (int)
        number of time windows to make
    window_width_units    : (str)
        units of the window_width arugment
        option 1: 'bins'
        option 2: 'seconds'
    even_steps           : (bool)
        if True, automatically spreads the windows over the entire time range
    custom_window_locs  : (numpy ndarray)
        if not None, then this be a 1D numpy ndarray with shape (numWindow, )
        these will be the start points each window, such that
        the window span will be start_point + window_width
        TODO: allow specifying the center of the windows instead of the starts
        Window location are specified in bin units.
    n_cv_splits : (int)
        number of cross validation splits
    n_repeats : (int)
        number of times to repeat the cross-validation evaluation
    random_shuffle_labels : (bool)
        if True, randomly shuffle labels provided (to act as a control)
    window_type : (str)
        type of window to make around each window location.
        option 1: 'forward', each window starts at window_loc and ends at window_loc + window_width
        option 2: 'center', each window is centered around window_loc with two sides adding up to window_width
    feature_importance_type : (str)
        type of data format to return the feature importance in
        option 1 : 'list': list of numpy arrays
        option 2 : 'df' pandas dataframe
    include_cell_loc : (bool)
        whether to include cell location in the feature importance dataframe
    scale_features : (bool)
        whether to scale features before running decoding
    include_confidence_score : (bool)
        applies to SVM only for now
        compute confidence score for each sample for being in the left or right
        for SVM, this is just the dot product between the weight vector and the sample
    TODO: num_bins should just be meta-data from activity_dataset / be inferred by the Time dimension shape
    TODO: tqdm for parallel processing jobs still not running properly

    Returns
    -------

    """

    if random_shuffle_labels is True:
        np.random.shuffle(labels)  # note that this is done in-place

    if (even_steps) is True and (custom_window_locs is None):
        window_start_locs = np.linspace(0, num_bins - window_width, num_window)
        window_end_locs = window_start_locs + window_width - 1  # due to the zero indexing
        # example: my window start at 0, and my width is 3, and so my window end loc is really 2
        # since 0, 1, 2 makes up three windows

        assert window_end_locs[-1] == num_bins - 1

    elif custom_window_locs is not None:
        if type(custom_window_locs) is int:
            custom_window_locs = np.array([custom_window_locs])
        elif type(custom_window_locs) is list:
            custom_window_locs = np.array(custom_window_locs)
        window_start_locs = custom_window_locs
        window_end_locs = window_start_locs + window_width - 1

    if clf is None:
        print('Warning: no classifier specified, using out-of-the-box linear SVM')
        clf = svm.SVC(kernel='linear')

    y = labels

    window_start_loc_list = list()
    window_end_loc_list = list()
    window_clf_accuracy_list = list()
    window_control_accuracy_list = list()
    repeated_cv_split_index_list = list()

    if extra_condition_activity_dataset is not None:
        window_clf_extra_condition_accuracy_list = list()

    classification_results_dict = defaultdict(list)

    if include_cell_loc:
        extra_field_dict = {'cell_loc': activity_dataset['CellLoc'].values}
    else:
        extra_field_dict = None

    if include_confidence_score:
        aud_cond = activity_dataset.isel(Cell=0)['audDiff'].values
        vis_cond = activity_dataset.isel(Cell=0)['visDiff'].values
    else:
        aud_cond = None
        vis_cond = None

    if run_parallel is True:
        # first create a list of feature matrix we are going to use

        feature_matrix_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(make_windowed_feature_matrix)(activity_dataset=activity_dataset, window_start_loc=win_start,
                                                  window_width=window_width, activity_name=activity_name,
                                                  window_width_units=window_width_units,
                                                  window_type=window_type,
                                                  ) for win_start in window_start_locs)

        if extra_condition_activity_dataset is not None:
            extra_feature_matrix_list = Parallel(n_jobs=-1, backend='threading')(
                delayed(make_windowed_feature_matrix)(activity_dataset=extra_condition_activity_dataset,
                                                      window_start_loc=win_start,
                                                      window_width=window_width, activity_name=activity_name,
                                                      window_width_units=window_width_units,
                                                      window_type=window_type) for win_start in window_start_locs)

        if extra_condition_activity_dataset is None:
            accuracy_score_list, window_control_accuracy_list, feature_importance_list, hyperparam_tune_list, \
            confidence_score_list = zip(
                *Parallel(n_jobs=-1, backend='threading')(delayed(predict_decision)(X=X, y=y, clf=clf,
                                                                                    control_clf=control_clf,
                                                                                    n_cv_splits=n_cv_splits,
                                                                                    n_repeats=n_repeats,
                                                                                    feature_importance_type=feature_importance_type,
                                                                                    extra_feature_fields=extra_field_dict,
                                                                                    get_importance=get_importance,
                                                                                    cv_split_method=cv_split_method,
                                                                                    tune_hyperparam=tune_hyperparam,
                                                                                    n_inner_loop_cv_splits=n_inner_loop_cv_splits,
                                                                                    n_inner_loop_repeats=n_inner_loop_repeats,
                                                                                    hyperparam_tune_method=hyperparam_tune_method,
                                                                                    param_grid=param_grid,
                                                                                    param_search_scoring_method=param_search_scoring_method,
                                                                                    include_confidence_score=include_confidence_score,
                                                                                    aud_cond=aud_cond,
                                                                                    vis_cond=vis_cond,
                                                                                    trial_cond=trial_cond) for X in
                                                          tqdm(feature_matrix_list, disable=(not print_loading_bar))))
            classification_results_dict['classifier_score'] = np.array(accuracy_score_list).flatten()
        else:
            accuracy_score_list, window_control_accuracy_list, feature_importance_list, hyperparam_tune_list, \
            confidence_score_list = zip(
                *Parallel(n_jobs=-1, backend='threading')(delayed(predict_decision)(X=X, y=y, clf=clf,
                                                                                    control_clf=control_clf,
                                                                                    n_cv_splits=n_cv_splits,
                                                                                    n_repeats=n_repeats,
                                                                                    feature_importance_type=feature_importance_type,
                                                                                    extra_feature_fields=extra_field_dict,
                                                                                    get_importance=get_importance,
                                                                                    cv_split_method=cv_split_method,
                                                                                    extra_X=X_extra,
                                                                                    extra_y=extra_labels,
                                                                                    tune_hyperparam=tune_hyperparam,
                                                                                    n_inner_loop_cv_splits=n_inner_loop_cv_splits,
                                                                                    n_inner_loop_repeats=n_inner_loop_repeats,
                                                                                    hyperparam_tune_method=hyperparam_tune_method,
                                                                                    param_grid=param_grid,
                                                                                    param_search_scoring_method=param_search_scoring_method,
                                                                                    include_confidence_score=include_confidence_score,
                                                                                    aud_cond=aud_cond,
                                                                                    vis_cond=vis_cond,
                                                                                    trial_cond=trial_cond) for
                                                          X, X_extra in
                                                          zip(feature_matrix_list, extra_feature_matrix_list)))

            # accuracy_score_list is now list of dicts
            original_data_accuracy_score = [x['original_condition'] for x in accuracy_score_list]
            extra_data_accuracy_score = [x['extra_condition'] for x in accuracy_score_list]
            classification_results_dict['classifier_score'] = np.array(original_data_accuracy_score).flatten()

            classification_results_dict['classifier_score_extra'] = np.array(extra_data_accuracy_score).flatten()

        # Note: repeat of [1, 2, 3] gives [1, 1, 2, 2, 3, 3], whereas tile give [1, 2, 3, 1, 2, 3]
        classification_results_dict['window_start_locs'] = np.repeat(window_start_locs, n_cv_splits * n_repeats)
        classification_results_dict['window_end_locs'] = np.repeat(window_end_locs, n_cv_splits * n_repeats)

        classification_results_dict['control_score'] = np.array(window_control_accuracy_list).flatten()
        classification_results_dict['repeated_cv_index'] = np.tile(np.arange(n_cv_splits * n_repeats),
                                                                   len(window_start_locs))

        # pdb.set_trace()
        classification_results_df = pd.DataFrame.from_dict(classification_results_dict)

        # Do the same for hyperparam tuning results: get the tuning result for each time window
        if tune_hyperparam:
            hyperparam_tune_results_list = list()
            # reminder: hyperparam_tune_list is a list of dataframes
            # hyperparam_tune_list has the same length as the number of windows
            for n_tune_df, tune_df in enumerate(hyperparam_tune_list):
                tune_df['window_start_locs'] = window_start_locs[n_tune_df]
                tune_df['window_end_locs'] = window_end_locs[n_tune_df]
                hyperparam_tune_results_list.append(tune_df)

        # TODO: add information about the window in the feature importance dataframe.

    else:

        hyperparam_tune_list = list()
        confidence_score_list = list()
        for window_start_loc, window_end_loc in zip(window_start_locs, window_end_locs):
            feature_matrix = make_windowed_feature_matrix(activity_dataset, window_start_loc=window_start_loc,
                                                          window_width=window_width, activity_name=activity_name,
                                                          window_width_units=window_width_units,
                                                          window_type=window_type)

            if extra_condition_activity_dataset is None:
                X_extra = None
                extra_labels = None
            else:
                X_extra = make_windowed_feature_matrix(extra_condition_activity_dataset,
                                                       window_start_loc=window_start_loc,
                                                       window_width=window_width, activity_name=activity_name,
                                                       window_width_units=window_width_units,
                                                       window_type=window_type)

                print(np.shape(X_extra))
                print(np.shape(extra_labels))

            accuracy_score_list, dummy_accuracy_score_list, feature_importance_list, hyperparam_tune_output, \
            confidence_score_output = predict_decision(X=feature_matrix, y=y,
                                                       clf=clf, control_clf=control_clf,
                                                       n_cv_splits=n_cv_splits,
                                                       n_repeats=n_repeats,
                                                       feature_importance_type=feature_importance_type,
                                                       extra_feature_fields=extra_field_dict,
                                                       get_importance=get_importance,
                                                       cv_split_method=cv_split_method,
                                                       extra_X=X_extra, extra_y=extra_labels,
                                                       tune_hyperparam=tune_hyperparam,
                                                       n_inner_loop_cv_splits=n_inner_loop_cv_splits,
                                                       n_inner_loop_repeats=n_inner_loop_repeats,
                                                       hyperparam_tune_method=hyperparam_tune_method,
                                                       param_grid=param_grid,
                                                       param_search_scoring_method=param_search_scoring_method,
                                                       include_confidence_score=include_confidence_score,
                                                       aud_cond=aud_cond, vis_cond=vis_cond,
                                                       trial_cond=trial_cond)

            # pdb.set_trace()

            # append to result lists
            hyperparam_tune_list.append(hyperparam_tune_output)
            confidence_score_list.append(confidence_score_output)

            if extra_condition_activity_dataset is None:
                accuracy_score_list = np.array(accuracy_score_list)

            dummy_accuracy_score_list = np.array(dummy_accuracy_score_list)

            if extra_condition_activity_dataset is not None:
                # accuracy_score_list is now list of dicts
                print(accuracy_score_list)
                original_data_accuracy_score = accuracy_score_list['original_condition']
                extra_condition_accuracy_score = accuracy_score_list['extra_condition']
                accuracy_score_list = original_data_accuracy_score

            window_start_loc_list.append(np.repeat(window_start_loc, len(accuracy_score_list)))

            window_end_loc_list.append(np.repeat(window_end_loc, len(accuracy_score_list)))
            repeated_cv_split_index_list.append(np.arange(len(accuracy_score_list)))
            window_clf_accuracy_list.append(accuracy_score_list)
            window_control_accuracy_list.append(dummy_accuracy_score_list)
            if extra_condition_activity_dataset is not None:
                window_clf_extra_condition_accuracy_list.append(extra_condition_accuracy_score)

        classification_results_dict['window_start_locs'] = np.concatenate(window_start_loc_list)
        classification_results_dict['window_end_locs'] = np.concatenate(window_end_loc_list)
        classification_results_dict['classifier_score'] = np.concatenate(window_clf_accuracy_list)
        classification_results_dict['control_score'] = np.concatenate(window_control_accuracy_list)
        classification_results_dict['repeated_cv_index'] = np.concatenate(repeated_cv_split_index_list)
        if extra_condition_activity_dataset is not None:
            classification_results_dict['extra_condition_classifier_score'] = \
                np.concatenate(window_clf_extra_condition_accuracy_list)

        classification_results_df = pd.DataFrame.from_dict(classification_results_dict)

        if tune_hyperparam:
            hyperparam_tune_results_list = list()
            # reminder: hyperparam_tune_list is a list of dataframes
            # hyperparam_tune_list has the same length as the number of windows
            for n_tune_df, tune_df in enumerate(hyperparam_tune_list):
                tune_df['window_start_locs'] = window_start_locs[n_tune_df]
                tune_df['window_end_locs'] = window_end_locs[n_tune_df]
                hyperparam_tune_results_list.append(tune_df)

    if include_peri_event_time:
        classification_results_df['window_start_sec'] = classification_results_df[
            'window_start_locs'].apply(
            lambda x: activity_dataset[peri_event_time_name].values[int(x)])

        classification_results_df['window_end_sec'] = classification_results_df[
            'window_end_locs'].apply(
            lambda x: activity_dataset[peri_event_time_name].values[int(x)])

        classification_results_df['window_width_sec'] = classification_results_df['window_end_sec'] - \
                                                        classification_results_df['window_start_sec']
    if include_baseline_accuracy:
        # Get baseline hit proportion if only predicting using the majority class
        baseline_accuracy = compute_baseline_performance(y=labels, metric='accuracy')
        classification_results_df['baseline_accuracy'] = baseline_accuracy

    # See whether feature importance output is needed
    if get_importance:
        if feature_importance_type == 'df':
            # Deal with having only a single dataframe (if only single window used)
            if type(feature_importance_list) == pd.core.frame.DataFrame:
                feature_importance_list = [feature_importance_list]
            df_list = list()
            for window_idx, feature_importance_df in enumerate(feature_importance_list):
                feature_importance_df['window'] = window_idx
                feature_importance_df['window_start_locs'] = window_start_locs[window_idx]
                feature_importance_df['window_end_locs'] = window_end_locs[window_idx]
                df_list.append(feature_importance_df)
            feature_importance_output = pd.concat(df_list)
        else:
            feature_importance_output = feature_importance_list
    else:
        feature_importance_output = None

    # See whether hyperparameter tuning output is needed
    if tune_hyperparam:
        # Output is a single dataframe
        hyperparam_tune_output = pd.concat(hyperparam_tune_results_list)
    else:
        hyperparam_tune_output = None

    if include_confidence_score:
        # Include window information (note that hyperparam result already did this early on)
        assert len(confidence_score_list) == len(window_start_locs)
        processed_confidence_df_list = list()
        for confidence_df, window_start_loc, window_end_loc in zip(
                confidence_score_list, window_start_locs, window_end_locs):
            confidence_df['window_start_locs'] = window_start_loc
            confidence_df['window_end_locs'] = window_end_loc

            processed_confidence_df_list.append(confidence_df)

        confidence_score_output = pd.concat(processed_confidence_df_list)
    else:
        confidence_score_output = None

    return classification_results_df, feature_importance_output, hyperparam_tune_output, confidence_score_output


def make_windowed_feature_matrix(activity_dataset, window_start_loc=0, window_width=20,
                                 window_center_loc=None,
                                 activity_name='firing_rate', window_width_units='bins',
                                 window_type='forward'):
    """
    Generates a feature matrix based on the average activity of each cell across a time window
    that is aligned to some stimulus or behaviour (eg. stimulus onset)
    Parameters
    ----------
    activity_dataset : xarray dataset
        xarray object with dimensions: Trial, Cell, and Time
    window_start_loc : int
        when the window should start (time bin) (for taking the mean)
    window_width     : int
        the length (in time bin units) of the window
    activity_name : str
        which variable to obtain from the activity_dataset
        (eg. 'firing_rate' or 'spike_count')
    window_type : str
        type of window to make around each window location.
        option 1: 'forward', each window starts at window_loc and ends at window_loc + window_width
        option 2: 'center', each window is centered around window_loc with two sides adding up to window_width
    Returns
    --------
    feature_matrix : numpy ndarray
        feature matrix with shape (num_trial, num_neurons)
    """

    if window_type == 'forward':
        window_end_loc = window_start_loc + window_width
    elif window_type == 'center':
        # window_start_loc = window_start_loc - (window_width / 2)
        # window_end_loc = window_start_loc + window_width
        if window_width_units == 'bins':
            assert window_width % 2 == 0, print('Window width needs to be even for centered windows with bin units')
        window_start_loc = window_center_loc - (window_width / 2)
        window_end_loc = window_center_loc + (window_width / 2)
    else:
        print('Warning: no valid window type specified.')

    if window_width_units == 'bins':
        mean_rate_feature_matrix = activity_dataset.sel(Time=slice(window_start_loc,
                                                 window_end_loc)).mean(dim='Time')
    else:
        print('Implement provision of window width in seconds.')

    if mean_rate_feature_matrix['Cell'].size > 1:
        mean_rate_feature_matrix = mean_rate_feature_matrix.transpose('Trial', 'Cell')
        feature_matrix = mean_rate_feature_matrix[activity_name].values
    else:
        feature_matrix = mean_rate_feature_matrix[activity_name].values
        feature_matrix = feature_matrix.reshape(-1, 1)

    return feature_matrix


def make_labels(decode_target, label_variable_name, all_brain_region_ds):
    """
    Make labels for doing classification
    Parameters
    ------------
    decode_target : str
        what to decode (on a trial by trial basis)
        'left_right' : decode left/right choice on each trial
        'audio_on_off' : decodes whether there is an auditory onset on each trial
        'audio_left_right' : decodes whether audio came from the left or right
        'visual_on_off' : decodes whether there is a visual stimuli on each trial
        'visual_left_right' : decodes whether visual stimuli came from the left or right
    label_variable_name :  str
        what is the name of the variable that contains information of decoding target
    all_brain_region_ds : xarray dataset
        xarray dataset with coordinates (Exp, Cell, Trial)
    :return:
    """
    # what to decode (usually responseMade means left/right or left/right/nogo)
    if decode_target == 'left_right':
        if 'Exp' in xr_util.get_ds_dim_names(all_brain_region_ds):
            labels = all_brain_region_ds[label_variable_name].isel(Exp=0, Cell=0).values.flatten()
        else:
            # some printing functions to check for NaNs (should replace with assert)
            # print(np.any(np.isnan(all_brain_region_ds['SpikeRate'].values)))
            labels = all_brain_region_ds[label_variable_name].isel(Cell=0).values.flatten()
            # print(np.any(np.isnan(labels)))
            print('Labels')
            print(labels)
            pdb.set_trace()

    elif decode_target == 'audio_on_off':
        # note that this accepts audio center as audio_on (unless you already subsetted the data
        # earlier)
        labels = np.isfinite(all_brain_region_ds['audDiff'].isel(Exp=0, Cell=0).values).astype(float)
    elif (decode_target == 'audio_left_right') or (decode_target == 'unimodal_audio_left_right'):
        # note that audio center and audio off conditions are removed

        assert len(np.unique(all_brain_region_ds['audDiff'])) <= 2, print(np.unique(all_brain_region_ds['audDiff']))
        # TimSit 2020-09-21: The above check just make sure that there are either (1) two audio levels, left and right
        # or (2) only one audio level, (left only or right only), in which case downstream code will skip the
        # decoding

        labels = all_brain_region_ds['audDiff'].isel(Exp=0, Cell=0).values > 0
        # -60: left, +60 right --> 0: left, 1:right
        labels = labels.astype(float)
    elif decode_target == 'visual_on_off':
        labels = (all_brain_region_ds['visDiff'].isel(Exp=0, Cell=0).values != 0) & \
                 (np.isfinite(all_brain_region_ds['visDiff'].isel(Exp=0, Cell=0).values))
        labels = labels.astype(float)
        # 0 : visual off, 1 : visual on
    elif (decode_target == 'visual_left_right') or (decode_target == 'unimodal_visual_left_right') or \
         (decode_target == 'unimodal_visual_left_right_0p8'):
        vis_diff_values = all_brain_region_ds['visDiff'].isel(Exp=0, Cell=0).values
        assert np.all(vis_diff_values != 0)  # double check only -ve (left) and +ve (right) values
        labels = (vis_diff_values < 0).astype(int)
    elif decode_target == 'go_no_go':
        # response made: 0 = timeout, 1 = left, 2 = right
        response_made = all_brain_region_ds[label_variable_name].isel(Exp=0, Cell=0).values.flatten()
        labels = (response_made >= 1).astype(int)
    elif decode_target == 'sensory_evidence':
        labels = anabehave.cal_sensory_evidence(behave_df=subset_behave_df,
                                                log_correction_term=0.0001)
    else:
        print('Warning: no valid decoding target selected')
        labels = None

    return labels


def make_subsampled_neurons(all_cell_alignment_data, target_brain_region='all',
                            min_neurons=30, random_seed=None,
                            reject_sub_threshold=True, custom_subsample_neurons=None,
                            verbose=False):
    """
    Subsample cells to make a fair comparison of decoding accuracy acaross brain regions.
    Number of subsample is the brain region with the smallest number of neurons.
    By default, brain regions with neurons fewer than min_neurons are removed.

    Parameters
    ----------
    all_cell_alignment_data : xarray dataset
        xarray dataset containing aligned neural activity with all brain regions
        should have dimensions: Cell, Exp, Time, Trial
    target_brain_region : str
        which brain region to obtain from all_cell_alignment_data datset
        if using the keyword 'all', then all brain regions are included
    min_neurons : int
        minimum number of neurons recorded in an experiment for that experiment to be included in the analysis
        if custom_subsample_neurons is set to None, then this is also the number of neurons to subsample
    random_seed : int
        rasndom seed for subsapmling neurons
    reject_sub_threshold : bool
        whether to return None if brain region contain fewer neurons than min_neurons
    verbose : bool
        whether to print messages about insufficient cell count
    Returns
    -------
    subsampled_cells : xarray dataset
        dataset with subsampled cells
        Should have dimensions: Cell, Exp, Time, Trials
    cell_count_dict : dict
        key: cell location (str), eg. 'FRP', 'MOs', 'ORBvl'
        value : number of cells (int)
    """
    unique_cell_loc = np.unique(all_cell_alignment_data['CellLoc'].values)
    cell_count_dict = dict()
    for cell_loc in unique_cell_loc:
        cell_count_dict[cell_loc] = len(all_cell_alignment_data.where(
            all_cell_alignment_data['CellLoc'] == cell_loc, drop=True)['CellLoc'].values)

    if custom_subsample_neurons is not None:
        num_subsample = custom_subsample_neurons
        min_neurons = custom_subsample_neurons
    else:
        cell_counts = np.array(list(cell_count_dict.values()))
        if np.max(cell_counts) < min_neurons:
            print('None of the brain region in this experiment exceeds min_neurons, returning None')
            return None, cell_count_dict
        else:
            num_subsample = np.min(cell_counts[cell_counts >= min_neurons])

    if target_brain_region == 'all':
        target_brain_loc_alignemnt_data = all_cell_alignment_data
    else:
        target_brain_loc_alignemnt_data = all_cell_alignment_data.where(
            all_cell_alignment_data['CellLoc'] == target_brain_region, drop=True
        )

    if reject_sub_threshold:
        if len(target_brain_loc_alignemnt_data['Cell'].values) < min_neurons:
            print("""Warning: brain region has fewer cells than threshold,
            returning None for first argument""")
            return None, cell_count_dict

    if random_seed is not None:
        np.random.seed(seed=random_seed)

    subsample_index = np.random.choice(
        target_brain_loc_alignemnt_data.Cell.values,
        size=num_subsample, replace=False)

    if verbose:
        print('Shape of subsample index')
        print(np.shape(subsample_index))

    subsampled_cells = target_brain_loc_alignemnt_data.sel(Cell=subsample_index)

    return subsampled_cells, cell_count_dict

def compute_baseline_performance(y, metric='accuracy'):
    """
    Computes baseline performance to be expected based on just knowing the labels (and nothing about the features)

    Parameters
    ----------
    y : numpy ndarray
        target categorical variable to be decoded, with shape (numObservations, ) or (numObservations, 1)
    metric : str
        how to calculate the performance metric
        options:
            'accuracy' : expected proportion of hits
    Returns
    -------
    baseline_performance : float
        the baseline performance to expect if guessing (using information only from labels)
    """

    if y.ndim >= 2:
        print('Warning: label has more than one dimension, will try squeezing it.')
        y = np.squeeze(y)

    if metric == 'accuracy':
        mode_value, mode_count = sstats.mode(y)
        prop_most_common_class = mode_count[0] / len(y)
        baseline_performance = prop_most_common_class

    return baseline_performance

