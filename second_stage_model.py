

import os
import itertools
import random

import numpy as np
import pandas as pd

from sklearn import linear_model, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plot


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

id_string = "485c0e2_2019_03_04_14_53_13"
lsms_out_path = os.path.join(base_path, "output/predict_{}.csv".format(id_string))

lsms_out = pd.read_csv(lsms_out_path, quotechar='\"',
                       na_values='', keep_default_na=False,
                       encoding='utf-8')

test_feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]

model_results_path = os.path.join(base_path, "output/models_{}.csv".format(id_string))


# plot.hist(lsms_out['ntl_2010'], bins=max(lsms_out['ntl_2010']), alpha=0.5, histtype='bar', ec='black')
# plot.xlabel('NTL')
# plot.ylabel('Frequency')
# plot.title('Histogram of NTL Values')
# plot.show()


# plot.hist(np.array([lsms_out[i] for i in test_feat_labels]).flatten())
# # plot.yscale("log")
# plot.xlabel('Features')
# plot.ylabel('Frequency')
# plot.title('Histogram of All Features Values')
# plot.show()


# -------------------------------------

'''
This script is to illustrate a solid cross validation process for this competition.
We use 10 fold out-of-bag overall cross validation instead of averaging over folds.
The entire process is repeated 5 times and then averaged.

You would notice that the CV value obtained by this method would be lower than the
usual procedure of averaging over folds. It also tends to have very low deviation.

Any scikit learn model can be validated using this.
'''


def R2(ypred, ytrue):
    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg)**2)
    SS_res = np.sum((ytrue - ypred)**2)
    r2 = 1 - (SS_res/SS_tot)
    return r2

def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    ypred = np.zeros((len(y), repeats))
    score = np.zeros(repeats)
    x = np.array(x)
    for r in range(repeats):
        i = 0
        # print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
        x, y = shuffle(x,y,random_state=r) #shuffle data before each repeat
        kf = KFold(n_splits=folds, random_state=i+1000) #random split, different each time
        for train_ind,test_ind in kf.split(x):
            # print('Fold', i+1, 'out of',folds)
            xtrain, ytrain = x[train_ind,:], y[train_ind]
            xtest, ytest = x[test_ind,:], y[test_ind]
            model.fit(xtrain, ytrain)
            ypred[test_ind, r] = model.predict(xtest)
            i += 1
        score[r] = R2(ypred[:, r], y)
    print 'Overall R2: {}'.format(score)
    print 'Mean: {}'.format(np.mean(score))
    print 'Deviation: {}'.format(np.std(score))
    return score

# -------------------------------------



def run_cv(X, y, k, k_inner, points, alpha_low, alpha_high, randomize=False):
    """
    Runs nested cross-validation to make predictions and compute r-squared.
    """
    alphas = np.logspace(alpha_low, alpha_high, points)
    r2s = np.zeros((k,))
    y_hat = np.zeros_like(y)
    kf = KFold(n_splits=y.size, shuffle=True).split(y_train)
    fold = 0
    for train_idx, test_idx in kf:
        r2s, y_hat, fold = evaluate_fold(
            X, y, train_idx, test_idx, k_inner, alphas, r2s, y_hat, fold,
            randomize)
    return y_hat, r2s.mean()


def evaluate_fold(
    X, y, train_idx, test_idx, k_inner, alphas, r2s, y_hat, fold,
        randomize):
    """
    Evaluates one fold of outer CV.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if randomize:
        random.shuffle(y_train)
    best_alpha = find_best_alpha(X_train, y_train, k_inner, alphas)
    X_train, X_test = scale_features(X_train, X_test)
    y_test_hat = train_and_predict_ridge(best_alpha, X_train, y_train, X_test)
    r2 = stats.pearsonr(y_test, y_test_hat)[0] ** 2
    r2s[fold] = r2
    y_hat[test_idx] = y_test_hat
    return r2s, y_hat, fold + 1


def find_best_alpha(X, y, k_inner, alphas):
    """
    Finds the best alpha in an inner CV loop.
    """
    kf = KFold(n_splits=k_inner, shuffle=True).split(y)
    best_alpha = 0
    best_r2 = 0
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kf:
            y_hat = predict_inner_test_fold(
                X, y, y_hat, train_idx, test_idx, alpha)
        r2 = stats.pearsonr(y, y_hat)[0] ** 2
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
    return best_alpha


def predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, alpha):
    """
    Predicts inner test fold.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)
    y_hat[test_idx] = train_and_predict_ridge(alpha, X_train, y_train, X_test)
    return y_hat


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    X_scaler = StandardScaler(with_mean=True, with_std=False)
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test


def train_and_predict_ridge(alpha, X_train, y_train, X_test):
    """
    Trains ridge model and predicts test set.
    """
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, y_train)
    y_hat = ridge.predict(X_test)
    return y_hat


def reduce_dimension(X, dimension):
    """
    Uses PCA to reduce dimensionality of features.
    """
    if dimension is not None:
        pca = PCA(n_components=dimension)
        X = pca.fit_transform(X)
    return X


y_train = lsms_out["cons"]
x_train_ntl = lsms_out[['ntl_2010']]
x_train_feat_all = lsms_out[test_feat_labels]

dimension = 100
x_train_feat = reduce_dimension(x_train_feat_all, dimension=dimension)

run_cv(x_train_ntl, y_train, k=5, k_inner=5, points=10, alpha_low=1, alpha_high=5, randomize=False)
run_cv(x_train_feat, y_train, k=5, k_inner=5, points=10, alpha_low=1, alpha_high=5, randomize=False)


# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
lm_list = [
    # "ARDRegression",
    # "TheilSenRegressor",
    # "BayesianRidge",
    # "ElasticNet",
    # "ElasticNetCV",
    # "LogisticRegression",
    # "LogisticRegressionCV",
    # "MultiTaskLasso",
    # "MultiTaskElasticNet",
    # "MultiTaskLassoCV",
    # "MultiTaskElasticNetCV",
    # "OrthogonalMatchingPursuitCV",
    # "PassiveAggressiveClassifier",
    # "Perceptron",
    # "RidgeClassifier",
    # "RidgeClassifierCV",
    # "SGDClassifier",

    {
        "name": "ridge",
        "model": "Ridge",
        "args": {}
    },{
        "name": "ridge_cv",
        "model": "RidgeCV",
        "args": {}
    },{
        "name": "ridge_cv5",
        "model": "RidgeCV",
        "args": {"cv": 5}
    },{
        "name": "ridge_cv10",
        "model": "RidgeCV",
        "args": {"cv": 10}
    },
    "Lars",
    "LarsCV",
    "LassoLars",
    "LassoLarsCV",
    "LinearRegression",
    "Lasso",
    # # "LassoCV",
    # "LassoLarsIC",
    # "HuberRegressor",
    # "OrthogonalMatchingPursuit",
    # "PassiveAggressiveRegressor",
    # "RANSACRegressor",
    # "SGDRegressor"
]

# https://scikit-learn.org/stable/modules/classes.html#regression-metrics

# metric_list = ["explained_variance_score", "mean_absolute_error", "median_absolute_error", "mean_squared_error", "mean_squared_log_error", "r2_score"]
# metric_abrv = ["evs", "mae", "mae2", "mse", "msle", "r2"]

metric_list = ["explained_variance_score", "mean_absolute_error", "median_absolute_error", "mean_squared_error", "r2_score"]
metric_abrv = ["evs", "mae", "mae2", "mse", "r2"]

keys = ["name", "model", "input"] + metric_abrv

results = []

print "Running models:"


# ==============================

metrics.r2_score(y_train, cross_val_predict(linear_model.Ridge(alpha=0.5), x_train_ntl, y_train, cv=2))
metrics.r2_score(y_train, cross_val_predict(linear_model.Ridge(alpha=0.5), x_train_feat, y_train, cv=2))

metrics.r2_score(y_train, linear_model.RidgeCV(cv=2, alphas=[0.5]).fit(x_train_ntl, y_train).predict(x_train_ntl))
metrics.r2_score(y_train, linear_model.RidgeCV(cv=2, alphas=[0.5]).fit(x_train_feat, y_train).predict(x_train_feat))

cross_validate(linear_model.Ridge(alpha=0.5), x_train_ntl, y_train, folds=2, repeats=1)
cross_validate(linear_model.Ridge(alpha=0.5), x_train_feat, y_train, folds=2, repeats=1)

# ==============================


for lm_dict in lm_list:

    if isinstance(lm_dict, str):
        name = model = lm_dict
        args = {}
    else:
        name = lm_dict["name"]
        lm = lm_dict["model"]
        args = lm_dict["args"]

    print "\t{}...".format(name)
    # get function corresponding to specified linear model
    lm_func = getattr(linear_model, lm)

    try:
        # run using NTL
        ntl_model = lm_func(**args)
        ntl_model.fit(x_train_ntl, y_train)
        ntl_preds = ntl_model.predict(x_train_ntl)
    except Exception as e:
        print (e)
        ntl_metric_vals = ["Error" for i in metric_list]
    else:
        ntl_metric_vals = [getattr(metrics, i)(y_train, ntl_preds) for i in metric_list]

    try:
        # run using CNN features
        cnn_model = lm_func(**args)
        cnn_model.fit(x_train_feat, y_train)
        cnn_preds = cnn_model.predict(x_train_feat)
    except Exception as e:
        print (e)
        cnn_metric_vals = ["Error" for i in metric_list]
    else:
        cnn_metric_vals = [getattr(metrics, i)(y_train, cnn_preds) for i in metric_list]


    results.append(dict(zip(keys, [name, lm, "ntl"] + ntl_metric_vals)))
    results.append(dict(zip(keys, [name, lm, "cnn"] + cnn_metric_vals)))






df = pd.DataFrame(results)
df = df[keys]
df.to_csv(model_results_path, index=False, encoding='utf-8')
