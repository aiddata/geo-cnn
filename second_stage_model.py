

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

id_string = "9a38491_2019_03_13_09_15_56" # resnet18 full fine tune batch 150
id_string = "baf2c1e_2019_03_13_11_46_24" # resnet50 full fine tune batch 150
id_string = "e310b22_2019_03_13_13_51_41" # resnet50 full fine tune batch 64
id_string = "4d91606_2019_03_13_15_09_48" # resnet152 full fine tune batch 64

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


# def cross_validate(model, x, y, folds=10, repeats=5):
#     '''
#     Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
#     model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
#     x = training data, numpy array
#     y = training labels, numpy array
#     folds = K, the number of folds to divide the data into
#     repeats = Number of times to repeat validation process for more confidence
#     '''
#     ypred = np.zeros((len(y), repeats))
#     score = np.zeros(repeats)
#     x = np.array(x)
#     for r in range(repeats):
#         i = 0
#         # print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
#         x, y = shuffle(x,y,random_state=r) #shuffle data before each repeat

#         kf = KFold(n_splits=folds, random_state=i+1000) #random split, different each time
#         for train_ind, test_ind in kf.split(y):
#             # print('Fold', i+1, 'out of',folds)
#             xtrain, ytrain = x[train_ind,:], y[train_ind]
#             xtest, ytest = x[test_ind,:], y[test_ind]
#             model.fit(xtrain, ytrain)
#             ypred[test_ind, r] = model.predict(xtest)

#             i += 1
#         score[r] = stats.pearsonr(ypred[:, r], y)[0] ** 2
#     print 'Overall R2: {}'.format(score)
#     print 'Mean: {}'.format(np.mean(score))
#     print 'Deviation: {}'.format(np.std(score))
#     return score


def pearson_r2(true, predict):
    return stats.pearsonr(true, predict)[0] ** 2


def run_cv(X, y, model, k, k_inner=5, alphas=None, metric=None, randomize=False):
    """
    Runs nested cross-validation to make predictions and compute r-squared.

    k_inner, alphas, metric only needed when determining ideal alpha param within folds

    (could add repeats to this, similar to demo cross_validate function above)

    """
    best_alpha = None
    y_true = []
    y_predict = []
    # score = np.zeros((k,))
    kf = KFold(n_splits=k, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(y)):
        # score, y_predict, fold = evaluate_fold(model, X, y, train_idx, test_idx, k_inner, alphas, score, y_predict, fold, randomize)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if randomize:
            random.shuffle(y_train)
        if alphas is not None:
            best_alpha = find_best_alpha(X_train, y_train, k_inner, model, metric, alphas)
        X_train, X_test = scale_features(X_train, X_test)
        y_test_predict = train_and_predict(X_train, y_train, X_test, model, best_alpha)
        y_true.append(y_test)
        y_predict.append(y_test_predict)
        # score[fold] = metric(y_test, y_test_predict)
    # return score.mean(), y_true, y_predict
    return y_true, y_predict


# def evaluate_fold(
#     model, X, y, train_idx, test_idx, k_inner, alphas, r2s, y_hat, fold, randomize):
#     """
#     Evaluates one fold of outer CV.
#     """
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     if randomize:
#         random.shuffle(y_train)

#     best_alpha = find_best_alpha(X_train, y_train, k_inner, model, alphas)
#     X_train, X_test = scale_features(X_train, X_test)
#     y_test_predict = train_and_predict(X_train, y_train, X_test, model, best_alpha)

#     r2 = stats.pearsonr(y_test, y_test_predict)[0] ** 2
#     r2s[fold] = r2
#     y_hat[test_idx] = y_test_predict
#     return r2s, y_hat, fold + 1


def find_best_alpha(X, y, k_inner, model, metric, alphas):
    """
    Finds the best alpha in an inner CV loop.
    """
    kfa = KFold(n_splits=k_inner, shuffle=True)
    best_alpha = 0
    best_score = 0
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kfa.split(y):
            y_hat = predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, model, alpha=alpha)
        score = metric(y, y_hat)
        if score > best_score:
            best_alpha = alpha
            best_score = score
    return best_alpha


def predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, model, alpha=None):
    """
    Predicts inner test fold.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)
    y_hat[test_idx] = train_and_predict(X_train, y_train, X_test, model, alpha=alpha)
    return y_hat


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    X_scaler = StandardScaler(with_mean=True, with_std=False)
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test


def train_and_predict(X_train, y_train, X_test, model, alpha=None):
    """
    Trains model and predicts test set.
    """
    if alpha:
        lm = model(alpha)
    else:
        lm = model()
    lm.fit(X_train, y_train)
    y_predict = lm.predict(X_test)
    return y_predict


# -------------------------------------


y_train = lsms_out["cons"].values
x_train_ntl = lsms_out[['ntl_2010']].values
x_train_cnn_all = lsms_out[test_feat_labels+['ntl_2010']].values

# reduce feature dimensions using PCA
pca_dimension = 15
pca = PCA(n_components=pca_dimension)
x_train_cnn = pca.fit_transform(x_train_cnn_all)


# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
lm_list = [
    {
        "name": "ridge",
        "model": linear_model.Ridge
    },{
        "name": "lasso",
        "model": linear_model.Lasso
    },{
        "name": "lassolars",
        "model": linear_model.LassoLars
    },{
        "name": "lars",
        "model": linear_model.Lars
    },{
        "name": "linear",
        "model": linear_model.LinearRegression
    },{
        "name": "ridge_cv10",
        "model": linear_model.Ridge,
        "k": 10,
        "k_inner": 10,
        # "alphas": [0.01, 0.1, 1, 5, 10],
        # "alphas": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20],
        "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },{
        "name": "lasso_cv10",
        "model": linear_model.Lasso,
        "k": 10,
        "k_inner": 5,
        "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },{
        "name": "lassolars_cv10",
        "model": linear_model.LassoLars,
        "k": 10,
        "k_inner": 5,
        "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },{
        "name": "lars_cv10",
        "model": linear_model.Lars,
        "k": 10
    },{
        "name": "linear_cv10",
        "model": linear_model.LinearRegression,
        "k": 10
    }
    # "LassoLarsIC",
    # "HuberRegressor",
    # "OrthogonalMatchingPursuit",
    # "PassiveAggressiveRegressor",
    # "RANSACRegressor",
    # "SGDRegressor"
]

# https://scikit-learn.org/stable/modules/classes.html#regression-metrics
metric_list = {
    "pr2": pearson_r2,
    "evs": metrics.explained_variance_score,
    "mae": metrics.mean_absolute_error,
    "mae2": metrics.median_absolute_error,
    "mse": metrics.mean_squared_error,
    "r2": metrics.r2_score
}

keys = ["name", "model", "input"] + metric_list.keys()

results = []

print "Running models:"


# ==============================


for lm_dict in lm_list:

    if "k" in lm_dict:

        name = lm_dict.pop("name")
        print "\t{}...".format(name)

        # run using NTL values
        try:
            ntl_y_true, ntl_y_predict = run_cv(x_train_ntl, y_train, **lm_dict)
        except Exception as e:
            print(e)
            ntl_metric_vals = ["Error" for i in metric_list]
        else:
            ntl_metric_vals = [np.array([metric_list[j](ntl_y_true[i], ntl_y_predict[i]) for i in range(lm_dict["k"])]).mean() for j in metric_list]

        # run using CNN features
        try:
            cnn_y_true, cnn_y_predict = run_cv(x_train_cnn, y_train, **lm_dict)
        except Exception as e:
            print(e)
            cnn_metric_vals = ["Error" for i in metric_list]
        else:
            cnn_metric_vals = [np.array([metric_list[j](cnn_y_true[i], cnn_y_predict[i]) for i in range(lm_dict["k"])]).mean() for j in metric_list]

    else:

        name = lm_dict["name"]
        print "\t{}...".format(name)
        lm_func = lm_dict["model"]

        # run using NTL values
        try:
            ntl_y_predict = train_and_predict(x_train_ntl, y_train, x_train_ntl, lm_func)
        except Exception as e:
            print (e)
            ntl_metric_vals = ["Error" for i in metric_list]
        else:
            ntl_metric_vals = [metric_list[i](y_train, ntl_y_predict) for i in metric_list]

        # run using CNN features
        try:
            cnn_y_predict = train_and_predict(x_train_cnn, y_train, x_train_cnn, lm_func)
        except Exception as e:
            print (e)
            cnn_metric_vals = ["Error" for i in metric_list]
        else:
            cnn_metric_vals = [metric_list[i](y_train, cnn_y_predict) for i in metric_list]


    results.append(dict(zip(keys, [name, lm_dict["model"].__name__, "ntl"] + ntl_metric_vals)))
    results.append(dict(zip(keys, [name, lm_dict["model"].__name__, "cnn"] + cnn_metric_vals)))



df = pd.DataFrame(results)
df = df[keys]
df.to_csv(model_results_path, index=False, encoding='utf-8')
