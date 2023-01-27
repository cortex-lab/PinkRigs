## kernel regression, adapted from Tim Sit

import numpy as np 
import pandas as pd
import collections,itertools
from utils.spike_dat import bincount2D
from utils.video_dat import digitise_motion_energy
from utils.plotting import off_axes

# import PinkRig utilities 
from Admin.csv_queryExp import load_data

# Machine learning / statistics
from sklearn.kernel_ridge import KernelRidge
import sklearn.linear_model as sklinear
import sklearn.metrics as sklmetrics
import sklearn.pipeline as sklpipe
import sklearn.model_selection as sklselection
import sklearn.dummy as skldummy
from sklearn.base import clone

import scipy.linalg as slinalg
import scipy.sparse as ssparse
from scipy.stats import zscore,median_abs_deviation

# for plotting default plots 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from utils.ev_dat import index_trialtype_perazimuth
import utils.plotting as pFT
from pylab import cm
from matplotlib.colors import LogNorm

#make toeplitz matrix 
def make_event_toeplitz(event_time_vector, event_bins,
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

    might change the way diag_value vector gets inputted
    """


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

    # good check if toeplitz is has to diag_value_vector, otherwise not 
    # if np.sum(full_event_toeplitz[:, -1]) != np.sum(event_time_vector):
    #     print('Warning: there are events that got truncated in '
    #             'peri-stimulus time bins')
    # assert np.sum(full_event_toeplitz[:, -1]) == np.sum(event_time_vector)

    return full_event_toeplitz
# make vectors of events
def make_vector(): 
    pass 

def test_toeplitz():
    pass
# make the feature matrix for every trial within the recording 
def make_feature_matrix(): 
    pass
#take the raw data and format it to feature matrix and corresponding data/spk (=target) matrix 
def format_data():
    pass
#or if we adapt tims then its a def. 
def evaluate_kernel_regression(): 
    pass

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

    def __init__(self, rank=10, reg=0, regressor=None, alpha=0, l1_ratio=1):
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
    #np.random.RandomState(seed).shuffle(unique) # stop shuffling so that the training set is always cv_number=0 
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


def fit_kernel_regression(X, Y, method='KernelRidge', fit_intercept=False, evaluation_method='fit-all', cv_split=10,
                          tune_hyper_parameter=True, split_group_vector=None, preprocessing_method=None,
                          rank=10, ridge_alpha=10, rr_regulariser=None, dev_test_random_seed=None, cv_random_seed=None, test_size=0.2,
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

    numerous parameters are in fact the starting parameters if a parameter search is performed

    rank 
    ridge_alpha etc. 

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
    fit_results = Bunch()

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
                print('ysd')
                # the column of 1s is already included in design matrix, so no need to fit intercept.
                clf = sklinear.Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept, solver='auto')
            elif method == 'nnegElasticNet':
                clf = sklinear.ElasticNet(alpha=1.0, fit_intercept=fit_intercept,
                                          positive=True)
            elif method == 'ReducedRankRegression':
                clf = ReducedRankRegressor(rank=50, reg=1.0)
            elif method == 'ReduceThenRidgeCV':
                clf = ReducedRankRegressor(rank=rank,alpha=ridge_alpha, regressor='Ridge', reg=rr_regulariser)
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
            print('evaluating via train-cv-test')

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
                model = sklinear.Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept, solver='auto')
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
                model = ReducedRankRegressor(regressor='Ridge',alpha=ridge_alpha,rank=rank,reg=rr_regulariser)
                param_grid = [{'rank': np.linspace(1, 200, 10).astype(int),
                               'alpha': np.logspace(-4, 4, num=10)}]
                scoring_method = sklmetrics.make_scorer(sklmetrics.r2_score,
                                                        multioutput='variance_weighted',
                                                        greater_is_better=True)
            elif method == 'ReduceThenElasticNetCV':
                model = ReducedRankRegressor(regressor='ElasticNet')
                param_grid = [{'rank': np.linspace(1, 50, 10).astype(int),
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
                               y_dev_prediction=y_dev_prediction,dev_inds=dev_inds,test_inds=test_inds,
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

    return reduced_X, reduced_Y, all_zero_rows,reduced_group_vector

def compute_response_function(model, event_names, event_start_ends, bin_width,event_start_index=1,
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

def test_kernel_significance(model, X, y, feature_column_dict,original_feature_column_dict=None,
                             method='leave-one-out', num_cv_folds=5,
                             split_group_vector=None,
                             sig_metric=['explained-variance'],
                             cv_random_seed=None, param_dict=None,
                             check_neg_features=False):
    """
    Test the significance of each kernel in the feature matrix.
    Parameters
    ----------------
    model  : (scikit-learn model class)
        sklearn model object with set hyperparameters (but not fitted to data yet)
        can also be a sklearn pipeline where hyperparameter tuning is done via cross validation
        within the training set.
    X : (numpy ndarray)
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
    original_feature_column_dict : used when the kernel is temporal and the feature column dict is a refurbished one 
    Returns
    --------------
    kernel_sig_results (pandas dataframe)
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

                print(event,train_idx.size,test_idx.size)

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
                    event_cv_df = pd.DataFrame.from_dict({'VE': explained_variance_score})

                if 'explained-variance-temporal' in sig_metric: 
                    # this is a temporal VE measure within trial to pull apart that from stimulus onset, how much VE is explained by the model in specific time bins. 
                    kernel_names = np.array(list(original_feature_column_dict.keys()))
                    trial_kernel_idx = [original_feature_column_dict[f].size>1 for f in original_feature_column_dict.keys()]
                    trial_kernel_names = kernel_names[trial_kernel_idx]
                    trial_rows_feature_matrix = [original_feature_column_dict[tk][:,np.newaxis] for tk in trial_kernel_names]
                    trial_rows_feature_matrix = np.concatenate(trial_rows_feature_matrix,axis=1)
                    # take the times of each
                    
                    n_trial_t_bins = trial_rows_feature_matrix.shape[0]
                    num_neuron  = test_diff.shape[1]
                    explained_variance_trial = np.zeros((n_trial_t_bins,num_neuron))
                    for i in range(n_trial_t_bins):
                        trial_time_index = (X_test[:,trial_rows_feature_matrix[i,:]]).sum(axis=1)>0
                        explained_variance_score = sklmetrics.explained_variance_score(test_diff[trial_time_index,:],
                                                                    one_kernel_test_prediction[trial_time_index,:],
                                                                    multioutput='raw_values')
                        explained_variance_trial[i,:] = explained_variance_score
                    if 'event_cv_df' in locals():
                        event_cv_df['VE_trial'] = [explained_variance_trial[:,n] for n in range(num_neuron)]
                    else:
                        event_cv_df = pd.DataFrame.from_dict({'VE_trial': [explained_variance_trial[:,n] for n in range(num_neuron)]})


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

def get_subselected_trials(events,trialtype,rt_min=None,rt_max=None,spl = None, contrast=None,vis_azimuth = None, aud_azimuth=None): 
    """
    helper function to select trial onsets based on criteria. 

    Parameters: 
    -----------------
    events: Bunch 
        requurement to pass on event bunch 
    trialtype: str
        indicating of the dict name in events that signifies the onset time 
        e.g. timeline_audPeriodOn 

    possible criteria of selection 
    ---------------------------------
    if all criteria is set to None, there is still subselection based on 
    ev.is_validTrial and whether the ev[trialtype] times were correctly extracted (i.e. drop nans)

    other possible criteria:

    rt_min
    rt_max 
    contrast
    vis_azimuth
    aud_azimuth 

    Returns: 
    --------------
        bool
        whether the trials in ev[trialtype] are selected or not based on criteria
    """
    ev_times = events[trialtype] 
    # at minimum subselect the valid tials that have a proper onset time
    is_selected = (~np.isnan(ev_times) & ~np.isinf(ev_times) & (events.is_validTrial==1))    
    if rt_min or rt_max:
        rt = events.timeline_choiceMoveOn - events.timeline_audPeriodOn
        if rt_min: 
            is_selected = is_selected & (rt>=rt_min)
        if rt_max:
            is_selected = is_selected & (rt<=rt_max)
        
    if contrast:
        is_selected = is_selected & ((events.stim_visContrast*100).astype('int')==contrast*100) 
    
    if spl:
        is_selected = is_selected & (np.round((events.stim_audAmplitude*100)).astype('int')==spl*100) 
    
    if vis_azimuth: 
        is_selected  = is_selected & (events.stim_visAzimuth==vis_azimuth)
    
    if aud_azimuth: 
        is_selected = is_selected & (events.stim_audAzimuth==aud_azimuth)

    return is_selected    


class events_list():
    """
    helper class to concatenate selected event times, their names and their called diag values
    """
    
    def __init__(self):
        self.times = []     
        self.names = [] # same length as the thing in ev_times
        self.fitted_trials_idx= [] 
        self.diag_values = {} # if the diag value is not 1 then the diag values dict should contain the ev_type_name and the corresponding vector. 


    def add_to_event_list(self,events,onset_sel_key,is_selected,feature_name_string,diag_values=[1]):
        """
        function to be able to easily add to lists I am concatenating 
        """
        ev_times_ = events[onset_sel_key][is_selected]
        onset_type_ = np.empty(ev_times_.size,dtype="object") 
        onset_type_[:] = feature_name_string

        self.times.append(ev_times_)        
        self.names.append(onset_type_) 
        self.fitted_trials_idx.append(np.where(is_selected)[0])

        if len(diag_values)!=len(ev_times_): 
            diag_values=np.ones(len(ev_times_))*diag_values
        
        self.diag_values[feature_name_string] = diag_values

    def finish_and_concat(self): 
        """
        after adding all the necessary information to lists, concatente them into a single array
       
        """
        self.times = np.concatenate(self.times)
        self.names = np.concatenate(self.names)
        self.fitted_trials_idx  = np.concatenate(self.fitted_trials_idx)



class kernel_model(): 
    def __init__(self,t_bin = 0.005, smoothing = 0.025):
        self.t_bin = t_bin
        self.smoothing = smoothing

    def load_and_format_data(self,event_types = ['vis','aud'],rt_params = None, subselect_neurons = None, probe='probe0',  
                            contrasts = [0.25], spls = [0.25],vis_azimuths = None, aud_azimuths = None,
                            t_support_stim = [-0.05,0.35],
                            t_support_movement =[-.2,0.1],
                            digitise_cam=False,
                            zscore_cam = False,
                            **kwargs):         

        """
        Parameters: 
        -------------------
        these parameters should determine which type of kernel is added 
        contrast: list: float
            determines which contrast values are used. If None all contrasts are equalised. 
        vis_azimuths: list/str
            if list: determines at which azimuths the kernels ought to be called at. Only applicable to onset kernels. 
            if None: only applicable to onset kernels: all azimuths are equalised 
            if str: 'dir' the L/R direction kernel ought to be called 
        aud_azimuths: list/str
            same as vis azimuths just for aud stimuli
        t_support_stim: list, len(list)=2
            time raange at which the stimulus kernels are supported 
        t_support_movement: list, len(list) =2
            time range at which the movement kernels are supported

        """

        self.digitise_cam = digitise_cam
        # load from PinkRigs pipeline

        data_dict = {'events':{'_av_trials':'table'},probe:{'spikes':['times','clusters']},'frontcam':{'camera':'all'},'eyeCam':{'camera':'all'}}

        recdat = load_data(data_name_dict = data_dict,**kwargs)

        if recdat.shape[0]!=1:
            print('several exps loaded: please be more specific')
        else: 
            print('successful loading.')
            print('binning events and spikes... This might take a while.')
            # prepare spike data  - bin and smooth 

            ################# SPIKE DATA ###########################
            spikes = recdat.iloc[0][probe].spikes  
            # subselect neurons 
            if subselect_neurons: 
                subselect_neurons = np.array(subselect_neurons)
                idx  = np.where(spikes.clusters==subselect_neurons[:,np.newaxis])[1]
                spikes.clusters = spikes.clusters[idx]  
                spikes.times = spikes.times[idx]      

            self.R,self.tscale,self.clusIDs = bincount2D(spikes.times,spikes.clusters,xbin=self.t_bin,xsmoothing=self.smoothing)

            
            ############## EVENTS DATA #########################

            # extract and digitise approproate events  
            ev = recdat.iloc[0].events._av_trials
            self.events  = ev

            if 'postactive' in recdat.expDef.iloc[0]:
                ev.is_validTrial = np.ones(ev.is_auditoryTrial.size)
            elif 'spatialIntegrationFlora' in recdat.expDef.iloc[0]:
                ev.is_validTrial = np.ones(ev.is_auditoryTrial.size)
            elif 'ckeckerboard_updatechecker' in recdat.expDef.iloc[0]:
                ev.is_validTrial = np.ones(ev.is_auditoryTrial.size)   

            extracted_ev = events_list()
            if 'vis' in event_types:   
                #onset_sel_key = 'timeline_visPeriodOn'  # timing info taken for this  
                onset_sel_key  = 'block_stimOn'
                azimuths = vis_azimuths
                if not contrasts: 
                    print('contrast is not passed. setting highest contrast only.')
                    contrasts = [np.max(ev.stim_visContrast)]     

                if not azimuths or 'dir' in azimuths: 
                    azimuths = [None]

                for my_contrast,my_azimuth in itertools.product(contrasts, azimuths): 
                    is_selected = get_subselected_trials(ev,onset_sel_key,contrast=my_contrast,vis_azimuth=my_azimuth,**rt_params)        
                    
                    feature_name_string = 'vis_kernel' + '_contrast_%.2f' % my_contrast
                    if my_azimuth:
                        feature_name_string += '_azimuth_%.0f' % my_azimuth

                    extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string)

                    if vis_azimuths:
                        if 'dir' in vis_azimuths: 
                            diag_value_vector = np.sign(ev.stim_visAzimuth[is_selected])
                            feature_name_string = feature_name_string + '_dir'
                            extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string,diag_values=diag_value_vector)

            if 'aud' in event_types:   
                #onset_sel_key = 'timeline_audPeriodOn'  # timing info taken for this  
                onset_sel_key  = 'block_stimOn'
                azimuths = aud_azimuths

                if not azimuths or 'dir' in azimuths: 
                    azimuths = [None]

                for my_spl,my_azimuth in itertools.product(spls,azimuths): 
                    is_selected = get_subselected_trials(ev,onset_sel_key,contrast=None,spl = my_spl, aud_azimuth=my_azimuth,**rt_params)        
                    
                    feature_name_string = 'aud_kernel' + '_spl_%.2f' % my_spl
                    if my_azimuth:
                        feature_name_string += '_azimuth_%.0f' % my_azimuth

                    extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string)

                    if aud_azimuths: 
                        if 'dir' in aud_azimuths: 
                            diag_value_vector = np.sign(ev.stim_audAzimuth[is_selected])
                            feature_name_string = feature_name_string + '_dir'
                            extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string,diag_values=diag_value_vector)

            if 'move' in event_types: 
                onset_sel_key = 'timeline_choiceMoveOn'  # timing info taken for this  

                is_selected = get_subselected_trials(ev,onset_sel_key,**rt_params)        
                    
                feature_name_string = 'move_kernel' 

                extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string)

                # add directionality by default
                diag_value_vector = np.sign(ev.timeline_choiceMoveDir[is_selected]-1.5) #just because LR is 1/2 in ev currently 
                feature_name_string = feature_name_string + '_dir'
                extracted_ev.add_to_event_list(ev,onset_sel_key,is_selected,feature_name_string,diag_values=diag_value_vector)  
    

            extracted_ev.finish_and_concat()

            self.events_digitised,_,event_names_ = bincount2D(extracted_ev.times,extracted_ev.names,xbin=self.t_bin,xlim = [np.min(self.tscale), np.max(self.tscale)])

            # create feature matrix 

            # get the different bin ranges for each event in event_digitised 
            stim_bin_range = np.arange(t_support_stim[0]/self.t_bin,t_support_stim[1]/self.t_bin).astype('int') 
            move_bin_range = np.arange(t_support_movement[0]/self.t_bin,t_support_movement[1]/self.t_bin).astype('int') 

            bin_ranges = {}
            for event_name in event_names_:
                if 'move' in event_name: 
                    bin_ranges[event_name]  = move_bin_range
                else:
                    bin_ranges[event_name] = stim_bin_range

            # make feature matrix from the digitised events
            toeplitz = [make_event_toeplitz(event_,bin_ranges[ev_name],diag_value_vector=extracted_ev.diag_values[ev_name]) for event_,ev_name in zip(self.events_digitised,event_names_)]
            toeplitz = np.concatenate(toeplitz,axis=1)


            self.feature_matrix = toeplitz

            # Add a first column diagonal of 1s
            #first_column = np.full(shape=(self.tscale.size, 1), fill_value=1)
            #self.feature_matrix = np.concatenate([first_column, feature_matrix],axis=1)

            kernel_lengths = [bin_ranges[ev_name].size for ev_name in event_names_]
            # duplicate for the directional kernels
            kernel_end_idx = np.insert(np.cumsum(kernel_lengths),0,0) 
            self.feature_column_dict = {}

            for ix,ev_name in enumerate(event_names_):
                self.feature_column_dict[ev_name] = np.array(range(kernel_end_idx[ix],kernel_end_idx[ix+1]))


            # add any kernels that are not fitted over time (baseline and camera)
            if 'move' not in event_types:
                trial_indices = [np.bitwise_and(self.tscale >= ts[0], self.tscale <= ts[-1]) for ts in zip(ev.block_stimOn+t_support_stim[0]-self.t_bin,ev.block_stimOn+t_support_stim[1]+.05)]
            else: 
                trial_indices = [np.bitwise_and(self.tscale >= ts[0], self.tscale <= ts[-1]) for ts in zip(ev.block_stimOn+t_support_stim[0],ev.timeline_choiceMoveOn+t_support_movement[1])]
                pass 
            trial_indices = np.concatenate(trial_indices).reshape((-1,self.tscale.size))
            fitted_trial_idxs = np.unique(extracted_ev.fitted_trials_idx)

            if 'baseline' in event_types:
                # add blank trials to bl kernel if requested?
                if 'blank' in event_types: 
                    #add blank trials to fitted trials too
                    blank_idxs =  np.where(ev.is_blankTrial)[0]
                    fitted_trial_idxs = np.unique(np.concatenate((blank_idxs,fitted_trial_idxs))) 

                                      
                bl_kernel = trial_indices[fitted_trial_idxs,:].sum(axis=0)
                # add a baseline to the feature matrix
                self.feature_matrix = np.concatenate((self.feature_matrix,bl_kernel[:,np.newaxis]),axis=1)
                self.feature_column_dict['baseline'] = np.array([self.feature_matrix.shape[1]-1])

            if 'motionEnergy' in event_types or 'pupil' in event_types:                 
                camtype_events = np.logical_or(['motionEnergy' in e for e in event_types], ['pupil' in e for e in event_types])                
                camtype_events = np.array(event_types)[camtype_events]

                self.cam_values = {}
                for camtype in camtype_events:
                    if 'motionEnergy' in camtype:
                        camera = recdat.iloc[0]['frontcam'].camera
                        cam_values = camera.ROIMotionEnergy

                        # this is not right as the the ROIAverageFrame is the frame size not the ROI size....
                        #n_pixels = camera.ROIAverageFrame.size 
                        #cam_values = cam_values/n_pixels  

                    elif 'pupil' in camtype: 
                        camera = recdat.iloc[0]['eyeCam'].camera
                        cam_values = camera.pupil_area_smooth    


                    # interpolate to the bins of the spiking
                    cam_values = np.interp(self.tscale,camera.times,cam_values)

                    # digitise the movement 
                    if digitise_cam:
                        _,_,cam_values = digitise_motion_energy(self.tscale,cam_values,plot_sample=True)
                    
                    if zscore_cam:
                        if 'zscore'in zscore_cam:
                            cam_values = zscore(cam_values)
                        elif 'mad' in zscore_cam: 
                            cam_values = (cam_values-np.median(cam_values))/median_abs_deviation(cam_values)

                    
                    self.cam_values[camtype] = cam_values
                    # add the movement to the feature matrix during the trial
                    kernel_idx  = trial_indices[fitted_trial_idxs,:].sum(axis=0).astype('bool')
                    cam_values[~kernel_idx] = 0
                    self.feature_matrix = np.concatenate((self.feature_matrix,cam_values[:,np.newaxis]),axis=1)
                    self.feature_column_dict[camtype] = np.array([self.feature_matrix.shape[1]-1]) 


            # creating training and test set for cross-validation - to do: balance trial types...
            np.random.seed(0)
            np.random.shuffle(fitted_trial_idxs)
            middle_index = fitted_trial_idxs.size//2

            
            train_idx = fitted_trial_idxs[:middle_index]
            test_idx = fitted_trial_idxs[middle_index:]


            train=(trial_indices[train_idx,:]*1).sum(axis=0).astype('int')
            test =(trial_indices[test_idx,:]*2).sum(axis=0).astype('int')


            self.split_group_vector=(train+test) # by definition to no timepoint can belong to two trials hence its ok to save to sum
            
            training_set= np.zeros(ev.is_blankTrial.size)         
            training_set[train_idx] = 1
            self.is_training_set = training_set.astype('bool')

            test_set= np.zeros(ev.is_blankTrial.size)         
            test_set[test_idx] = 1
            self.is_test_set = test_set.astype('bool')      

    def fit(self,**fit_kwargs):   
        feature_matrix_,R_,a,split_group_vector_=remove_all_zero_rows(self.feature_matrix,self.R.T,group_vector=self.split_group_vector) 

        self.fit_results = fit_kernel_regression(
            feature_matrix_,R_,
            evaluation_method='train-cv-test',cv_split=2,
            split_group_vector=split_group_vector_,**fit_kwargs
            ) 

    def evaluate(self,kernel_selection = 'independent',sig_metric = ['explained-varaince']):
        """
        independent evaluation function using various modes
        Parameters: 
        ----------- 
        kernel_selection: str
            implement grouping of kernels


        """

        feature_matrix_,R_,a,split_group_vector_=remove_all_zero_rows(self.feature_matrix,self.R.T,group_vector=self.split_group_vector) 
        model=clone(self.fit_results['test_model'])

        if 'stimgroups' in kernel_selection: 
            kernel_names = list(self.feature_column_dict.keys())
            # merge all aud or vis groups 
            new_feature_column_dict = {}
            new_feature_column_dict['aud'] = np.concatenate([self.feature_column_dict[k] for k in kernel_names if 'aud' in k])
            new_feature_column_dict['vis'] = np.concatenate([self.feature_column_dict[k] for k in kernel_names if 'vis' in k])  
            remaining_kernels = [k for k in kernel_names if ('aud' not in k) and ('vis' not in k)]            
            for k in remaining_kernels: 
                new_feature_column_dict[k] = self.feature_column_dict[k]

            kernel_significance = test_kernel_significance(
                model,feature_matrix_,R_,
                new_feature_column_dict,original_feature_column_dict = self.feature_column_dict,
                num_cv_folds=2, 
                sig_metric=sig_metric,
                split_group_vector=split_group_vector_
                )            

        else: 
            kernel_significance = test_kernel_significance(
                model,feature_matrix_,R_,
                self.feature_column_dict,num_cv_folds=2, 
                sig_metric=sig_metric,
                split_group_vector=split_group_vector_
                ) 

        kernel_significance['clusID'] = [self.clusIDs[n] for n in kernel_significance.neuron]

        print('kernel_signifiance_is_ready')

        return kernel_significance   

    def fit_evaluate(self,get_prediciton=True,**fit_kwargs): 
        """
        Fit the kernel regression 
        Fitting procedure: get rid of zero-rows: ie. rows where no kernel is activated 


        Parameters:
        ------------
        get_prediction: bool
            whether to get prediciton for the entire feature matrix (with the zero rows)
        
        """

        feature_matrix_,R_,a,split_group_vector_=remove_all_zero_rows(self.feature_matrix,self.R.T,group_vector=self.split_group_vector)

        print(np.unique(split_group_vector_))

        # # add the first column to the fea ture matrix if requested for baseline firing rate
        # first_column_ = np.full(shape=(feature_matrix_.shape[0], 1), fill_value=1)
        # feature_matrix_ = np.concatenate([first_column_, feature_matrix_],axis=1)

        # # add that to the original feature matrix too? 
        # first_column = np.full(shape=(self.feature_matrix.shape[0], 1), fill_value=1)
        # self.feature_matrix = np.concatenate([first_column, self.feature_matrix],axis=1)

        self.fit_results = fit_kernel_regression(
            feature_matrix_,R_,
            evaluation_method='train-cv-test',cv_split=2,
            split_group_vector=split_group_vector_,**fit_kwargs
            ) 

        model=clone(self.fit_results['test_model'])
        self.kernel_significance = test_kernel_significance(
            model,feature_matrix_,R_,
            self.feature_column_dict,num_cv_folds=2, 
            sig_metric=['explained-variance'],
            split_group_vector=split_group_vector_
            )

        if get_prediciton:
            predict_all = self.fit_results['dev_model'].predict(self.feature_matrix)
            self.prediction = predict_all.T

        
    

    def save_model(): 
        """
        implement saving of the entire class. that can be reloaded refitted etc. 
        """
        pass 
    

    def get_raster(self,myev_times,t_before = 0.2, t_after = .2, sort_idx = None, spike_type = 'data',sortAmp = False):
        """
        return raster of events 

        Parameters: 
        ----------------
        myev_times = numpy.ndarray
            times in s for the event times around which the raster is requested to be aliged
        t_before: float 
            time in s, before events +ve value!
        t_after: float 
            time in s, after events
        sort_idx: numpy ndarray
            same length as myev_times if raster is reqested to be sorted. 
        spike_type: str
            either equals data or prediction 

        Returns:
        ----------
        rasters: numpy ndarray
            nrn x time
        tscale: numpy ndarray (time,)
            timescale in s of the 2nd dim of the raster
        sort_idx: str (None)/numpy nd array
            sorting index of myev_times if sorted based on PC1/amp etc.

        Todo: 
        ---------
        Implement neuronID = None,or other specific requests for neurons 

        """
        myev_digi,_,_ = bincount2D(myev_times,np.zeros(myev_times.size).astype('int'),xbin=self.t_bin,xlim = [np.min(self.tscale), np.max(self.tscale)])
        bin_range = np.arange(-t_before/self.t_bin,t_after/self.t_bin).astype('int') 
        zero_bin_idx = np.argmin(np.abs(bin_range))
        onset_idx = np.where(myev_digi[0,:]==1)[0]
        if sort_idx is not None: 
            onset_idx = onset_idx[sort_idx]

        bin2show  = [(bin_range + idx)[np.newaxis,:] for idx in onset_idx]
        bin2show = np.concatenate(bin2show,axis=0)

        if 'data' in spike_type: 
            raster = self.R[:,bin2show]
        elif 'pred' in spike_type:
            raster = self.prediction[:,bin2show]
        elif spike_type in self.cam_values.keys():
            raster = self.cam_values[spike_type][bin2show]

        # to do implement sorting of raster by amp
        if sortAmp:
            sort_idx=np.argsort(raster[:,zero_bin_idx:].mean(axis=1))
            raster = raster[sort_idx,:]


        return Bunch({'raster': raster,'tscale': bin_range*self.t_bin,'sort_idx': sort_idx}) 


    def plot_pred_helper(self,on_times,is_sel,nrnID_idx,ax,raster_kwargs=None,c='k',plot_train = True, plot_test= False, plot_pred = True):

        if not raster_kwargs: 
            raster_kwargs = {
                't_before': 0.2, 
                't_after': 0.4, 
                'sort_idx': None
            } 

        if plot_train or plot_pred: 
            trial_set = self.is_training_set
        elif plot_test: 
            trial_set = self.is_test_set 
        
        on_time = on_times[(trial_set & is_sel)]

        if on_time.size>2:
            if plot_train or plot_test:
                dat = self.get_raster(on_time,spike_type = 'data',**raster_kwargs)
                bin_range = dat.tscale
                dat = dat.raster[nrnID_idx,:,:]                      

                mean = dat.mean(axis=0)
                bars = dat.std(axis=0)/dat.shape[0]
                ax.fill_between(bin_range, mean - bars, mean + bars, color=c,alpha=.4)

            if plot_pred: 
                pred = self.get_raster(on_time,spike_type = 'pred',**raster_kwargs)
                bin_range = pred.tscale
                pred = pred.raster[nrnID_idx,:,:]
                ax.plot(bin_range,pred.mean(axis=0),color=c)


    def plot_prediction(self,nrnID,plot_stim = True, plot_move=False, sep_choice=True, **plot_cond_kwargs):
        
        if plot_stim & plot_move: 
            fig,ax=plt.subplots(3,6,figsize=(20,10),sharex=True,sharey=True) 
            stim_plot_inds = np.array([0,2,4])
            move_plot_inds = np.array([1,3,5])
        elif plot_stim and not plot_move:
            fig,ax=plt.subplots(3,3,figsize=(10,10),sharex=True,sharey=True)    
            stim_plot_inds = np.array([0,1,2])   
        elif plot_move and not plot_stim:
            fig,ax=plt.subplots(3,3,figsize=(10,10),sharex=True,sharey=True)    
            move_plot_inds = np.array([0,1,2])     

        fig.patch.set_facecolor('xkcd:white')

        nrnID_idx = np.where(self.clusIDs==nrnID)[0][0]
        if nrnID_idx.size!=1: 
            print('neuron not fitted or not found')

        mycolors = ['blue','red']        
        plotted_vis_azimuth = [-60,0,60]
        plotted_aud_azimuth = [60,0,-60]
        vazi,aazi=np.meshgrid(plotted_vis_azimuth,plotted_aud_azimuth)
        for i,m in enumerate(vazi):
            for j,_ in enumerate(m):
                v = vazi[i,j]
                a = aazi[i,j]
                trialtype=index_trialtype_perazimuth(a,v)

                if 'aud' in trialtype:
                    visazimcheck =np.isnan(self.events.stim_visAzimuth)
                elif 'blank' in trialtype:
                    visazimcheck =np.isnan(self.events.stim_visAzimuth)
                else:
                    visazimcheck = (self.events.stim_visAzimuth==v)

                if sep_choice:
                    n_lines = 2
                else: 
                    n_lines = 1

                for mydir in range(n_lines):                  

                    is_selected_trial = (self.events[trialtype]==1) & (self.events.stim_audAzimuth==a)  & visazimcheck 
                    
                    if sep_choice:
                        is_selected_trial = is_selected_trial & (self.events.timeline_choiceMoveDir==mydir+1)

                    if plot_stim:                        
                        myax = ax[i,stim_plot_inds[j]]
                        rkw = {'t_before': 0.05,'t_after': 0.8,'sort_idx': None}
                        self.plot_pred_helper(self.events.timeline_audPeriodOn,is_selected_trial,nrnID_idx,myax,
                                            c=mycolors[mydir],raster_kwargs= rkw,
                                            **plot_cond_kwargs)
                        myax.axvline(0, color ='k',alpha=0.7,linestyle='dashed')

                        pFT.off_axes(myax)
                        myax.set_ylabel(a)
                        myax.set_xlabel(v)       

                    if plot_move:
                        myax = ax[i,move_plot_inds[j]]
                        rkw = {'t_before': 0.05,'t_after': 0.8,'sort_idx': None}
                        self.plot_pred_helper(self.events.timeline_choiceMoveOn,is_selected_trial,nrnID_idx,myax,
                                            c=mycolors[mydir],raster_kwargs=rkw,
                                            **plot_cond_kwargs)

                        myax.axvline(0, color ='k',alpha=0.7,linestyle='dashed')
                        pFT.off_axes(myax)              
                    

                    ax[2,-1].hlines(-0.1,0.25,0.35,'k')

    def plot_prediction_rasters(self,nrnID,raster_kwargs = None ,visual_azimuth = None, auditory_azimuth = None, contrast = 1, spl = .1): 
        """
        this fuction plots the prediction in a raster format

        Parameters: 
        ------------
        nrnID: float
            cluster ID of neuron plotted
        raster_kwargs: dict/None
            paramters of raster, can include t_before, t_after and sort_idx
        visual_azimuth: list 
        auditory azimuth: list, same length as visual azimuth. 
            if unisensory trials, azimuth request ought to be NaN 
        contrast/spl: float 
            can only call one contrast and spl  for now 
                 

        """
        # calculate index of neuron from the ID number
        nrnID_idx = np.where(self.clusIDs==nrnID)[0][0]

        if not raster_kwargs:
           raster_kwargs = {'t_before': 0,'t_after': 0.6,'sort_idx': None} 

        # if len(visual_azimuth)==1: 
        #     ax = ax [:,np.newaxis]

        if (visual_azimuth is None) & (auditory_azimuth is None):
            on_time = self.events.timeline_audPeriodOn[~np.isnan(self.events.timeline_audPeriodOn)]
            fig,ax = plt.subplots(1,3,figsize=(4,12),sharex=True)
            if plot_cam:
                self.cam_values.keys()
                raster_kwargs['sort_idx'] = None
                cam = self.get_raster(on_time,spike_type = 'cam',sortAmp=True,**raster_kwargs)
                raster_kwargs['sort_idx'] = cam.sort_idx
                ax[0].matshow(cam.raster, cmap=cm.gray_r, norm=LogNorm(vmin=-1, vmax=5)) 

            dat = self.get_raster(on_time,spike_type = 'data', **raster_kwargs)
            pred = self.get_raster(on_time,spike_type = 'pred', **raster_kwargs)

            ax[1].matshow(dat.raster[nrnID_idx,:,:], cmap = 'Greys')
            ax[2].matshow(pred.raster[nrnID_idx,:,:], cmap = 'Greys')

        else:
            print('azimuthal separation')
            # plotting the data anad the prediction in different columns 

            no_plots = 2
            if 'cam_values' in dir(self):
                no_plots += len(self.cam_values)
                plot_cam = True
            else: 
                plot_cam = False

            fig,ax = plt.subplots(no_plots,len(visual_azimuth),figsize=(16,12),sharex=True)   

            for idx,(v,a) in enumerate(zip(visual_azimuth,auditory_azimuth)): 
                if np.isnan(v): 
                    is_called_vis = np.isnan(self.events.stim_visAzimuth)
                else:
                    is_called_vis = (self.events.stim_visAzimuth == v) & ((self.events.stim_visContrast*100).astype('int')==contrast*100) 

                if np.isnan(a):    
                    is_called_aud = np.isnan(self.events.stim_audAzimuth)
                else:
                    is_called_aud = (self.events.stim_audAzimuth == a) & (np.round((self.events.stim_audAmplitude*100)).astype('int')==spl*100)  

                if ~np.isnan(a):
                    on_time = self.events.timeline_audPeriodOn[(is_called_vis & is_called_aud)]
                else: 
                    on_time = self.events.timeline_visPeriodOn[(is_called_vis & is_called_aud)]


                if plot_cam:

                    for n_cam_idx,camname in enumerate(self.cam_values.keys()):                        
                        if n_cam_idx==0:
                            raster_kwargs['sort_idx'] = None
                            cam = self.get_raster(on_time,spike_type = camname,sortAmp=True,**raster_kwargs)
                            raster_kwargs['sort_idx'] = cam.sort_idx
                        
                        else:
                            cam = self.get_raster(on_time,spike_type = camname,sortAmp=False,**raster_kwargs)
                            cam.raster = cam.raster[raster_kwargs['sort_idx'],:]

                        if self.digitise_cam:
                            ax[n_cam_idx,idx].matshow(cam.raster,cmap='Greys',vmin=0,vmax=1) # for the digitised movement                    
                        else:
                            #ax[idx,n_cam_idx].matshow(cam.raster, cmap=cm.gray_r, norm=LogNorm(vmin=np.min(cam.raster), vmax=np.max(cam.raster))) 
                            ax[n_cam_idx,idx].matshow(cam.raster, cmap=cm.gray_r, vmin=np.min(cam.raster), vmax=np.max(cam.raster))
                    
                    


                dat = self.get_raster(on_time,spike_type = 'data', **raster_kwargs)
                pred = self.get_raster(on_time,spike_type = 'pred', **raster_kwargs)

                ax[-2,idx].matshow(dat.raster[nrnID_idx,:,:], cmap = 'Blues',vmin = np.min(dat.raster), vmax=np.max(dat.raster)/4)
                ax[-2,idx].set_title('vis: %.0f, aud %.0f' % (v,a))
                ax[-1,idx].matshow(pred.raster[nrnID_idx,:,:], cmap = 'Blues',vmin = np.min(dat.raster), vmax=np.max(dat.raster)/4)
                off_axes(ax[-2,idx])
                off_axes(ax[-1,idx])

        fig.show()
      
    def plot_kernels(self,nrnID):
        nrn_idx = np.where(self.clusIDs==nrnID)[0] # 

        model = self.fit_results['test_model']

        if type(model).__name__=='ReducedRankRegressor':
            fits = np.dot(model.regressor_model.coef_,model.A)
        else: 
            fits = model.coef_

        
        test_ve = self.kernel_significance[(self.kernel_significance.neuron==nrn_idx[0]) & (self.kernel_significance.cv_number==0)]

        n_kernels = len(list(self.feature_column_dict.keys())) 
        plt.rcParams.update({'font.size':8})
        fig,ax = plt.subplots(1,n_kernels,figsize=(20,4),sharey=True)
        fig.patch.set_facecolor('xkcd:white')

        for i,feature in enumerate(self.feature_column_dict):
            kernel = np.ravel(fits[nrn_idx,self.feature_column_dict[feature]])
            test_ve_kernel = test_ve['VE'][test_ve.event==feature].values[0]
            if 'Dir' in feature: 
                ax[i].plot(kernel,'orange')#-fits[nrnID,0])
            else: 
                if kernel.size==1: 
                    ax[i].text(0,0,'%.2f' % kernel)
                else: 
                    ax[i].plot(kernel,'k')
            ax[i].set_title('%s \n VE,test = %.1f%%' % (feature,test_ve_kernel*100),rotation=45)

            pFT.off_axes(ax[i])

        fig.suptitle(nrnID)