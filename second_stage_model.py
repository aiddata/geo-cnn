

import os
import glob
import itertools
import random
import time
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn import linear_model, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plot

from settings_builder import Settings

# -----------------------------------------------------------------------------


# *****************
# *****************
json_path = "settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


s = Settings()
s.load(json_path)
base_path = s.base_path

mode = s.config["second_stage_mode"]

model_tag = s.config["model_tag"]

predict_hash = s.build_hash(s.data[s.config["predict"]], nchar=7)

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

# merge_out_path = os.path.join(base_path, "output/models_merge_{}_{}.csv".format(timestamp, model_tag))

# -----------------

regex_str = os.path.join(base_path, "output/s1_predict/predict_*_{}_{}_{}.csv".format(predict_hash, s.config["version"], s.config["predict_tag"]))
regex_search = glob.glob(regex_str)

qlist = ["_".join(os.path.basename(i).split("_")[1:])[:-4] for i in regex_search]

print qlist

# qlist = ["7a118a3_2019_03_28_12_48_37"]

# qlist = pd.read_csv(os.path.join(base_path, "cnn_results_merge_1.csv"))["id_string"].tolist()

# -------------------------------------


def pearson_r2(true, predict):
    return stats.pearsonr(true, predict)[0] ** 2


# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
model_lookup = {
    "ridge": {
        "model": linear_model.RidgeCV
    },
    "lasso": {
        "model": linear_model.Lasso
    },
    "lassolars": {
        "model": linear_model.LassoLars
    },
    "lars": {
        "model": linear_model.Lars
    },
    "linear": {
        "model": linear_model.LinearRegression
    },
    "ridge-cv10": {
        "model": linear_model.Ridge,
        "k": 10,
        "k_inner": 10,
        # "alphas": [0.01, 0.1, 1, 5, 10],
        "alphas": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20],
        # "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },
    "lasso-cv10": {
        "model": linear_model.Lasso,
        "k": 10,
        "k_inner": 5,
        "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },
    "lassolars-cv10": {
        "model": linear_model.LassoLars,
        "k": 10,
        "k_inner": 5,
        "alphas": np.logspace(0.5, 10, 10),
        "metric": pearson_r2
    },
    "lars-cv10": {
        "model": linear_model.Lars,
        "k": 10
    },
    "linear-cv10": {
        "model": linear_model.LinearRegression,
        "k": 10
    }
    # "LassoLarsIC",
    # "HuberRegressor",
    # "OrthogonalMatchingPursuit",
    # "PassiveAggressiveRegressor",
    # "RANSACRegressor",
    # "SGDRegressor"
}


# https://scikit-learn.org/stable/modules/classes.html#regression-metrics
metric_lookup = {
    "pr2": pearson_r2,
    "evs": metrics.explained_variance_score,
    "mae": metrics.mean_absolute_error,
    "mae2": metrics.median_absolute_error,
    "mse": metrics.mean_squared_error,
    "r2": metrics.r2_score
}

metric_list = {i:metric_lookup[i] for i in s.data["second_stage"]["metrics"]}

keys = ["id", "name", "model", "input"] + metric_list.keys()


# -------------------------------------

def plot_ntl(ntl_data):
    plot.hist(ntl_data, bins=max(ntl_data), alpha=0.5, histtype='bar', ec='black')
    plot.xlabel('NTL')
    plot.ylabel('Frequency')
    plot.title('Histogram of NTL Values')
    plot.show()


def plot_cnn_feats(feat_data, feat_labels):
    plot.hist(np.array([feat_data[i] for i in feat_labels]).flatten())
    # plot.yscale("log")
    plot.xlabel('Features')
    plot.ylabel('Frequency')
    plot.title('Histogram of All Features Values')
    plot.show()

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


def train(X_train, y_train, model, alpha=None):
    """
    Trains model and predicts test set.
    """
    if alpha:
        lm = model(alpha)
    else:
        lm = model()
    lm.fit(X_train, y_train)
    return lm


def train_and_predict(X_train, y_train, X_test, model, alpha=None):
    """
    Trains model and predicts test set.
    """
    lm = train(X_train, y_train, model, alpha=None)
    y_predict = lm.predict(X_test)
    return y_predict



# -------------------------------------


def run(id_string):
    """
    id_string = <train hash>_<predict hash>_<version tag>_<predict tag>
    """
    pred_data_path = os.path.join(base_path, "output/s1_predict/predict_{}.csv".format(id_string))

    pred_data = pd.read_csv(pred_data_path, quotechar='\"',
                        na_values='', keep_default_na=False,
                        encoding='utf-8')

    test_feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]



    # reduce feature dimensions using PCA
    pca_dimension = 15
    pca = PCA(n_components=pca_dimension)

    y_train = pred_data["pred_yval"].values

    x_train = {}
    x_train["ntl"] = pred_data[['ntl']].values
    x_train["cnn"] = pred_data[test_feat_labels].values

    x_train["all"] = x_train["ntl"] + x_train["cnn"]
    x_train["all_pca{}".format(pca_dimension)] = pca.fit_transform(x_train["ntl"] + x_train["cnn"])

    x_train["cnn_pca{}".format(pca_dimension)] = pca.fit_transform(x_train["cnn"])
    x_train["cnn_pca{}_ntl".format(pca_dimension)] = x_train["cnn_pca{}".format(pca_dimension)] + x_train["ntl"]


    print "Running models:"


    for name in s.data["second_stage"]["models"]:

        lm_dict = deepcopy(model_lookup[name])

        results = []

        models_results_path = os.path.join(base_path, "output/s2_models/models_{}_{}_{}.joblib".format(name, id_string, model_tag))
        metrics_results_path = os.path.join(base_path, "output/s2_metrics/metrics_{}_{}_{}.csv".format(name, id_string, model_tag))

        for x_name, x_data in x_train.iteritems():

            print "\t{}({})...".format(name, x_name)

            lm = train(x_data, y_train, lm_dict["model"])
            joblib.dump(lm, models_results_path)

            # run with or without cross validation
            if "k" in lm_dict:

                try:
                    y_true, y_predict = run_cv(x_data, y_train, **lm_dict)
                except Exception as e:
                    print(e)
                    metric_vals = ["Error" for i in metric_list]
                else:
                    metric_vals = [
                        np.array([metric_list[j](y_true[i], y_predict[i])
                        for i in range(lm_dict["k"])]).mean() for j in metric_list
                    ]

            else:

                try:
                    y_predict = lm.predict(x_data)
                except Exception as e:
                    print(e)
                    metric_vals = ["Error" for i in metric_list]
                else:
                    metric_vals = [metric_list[i](y_train, y_predict) for i in metric_list]

            tmp_vals = [id_string, name, lm_dict["model"].__name__, x_name]
            tmp_vals += metric_vals
            results.append(dict(zip(keys, tmp_vals)))

        df = pd.DataFrame(results)
        df = df[keys]
        df.to_csv(metrics_results_path, index=False, encoding='utf-8')


# -----------------------------------------------------------------------------

rank = 0
if mode == "parallel":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


if mode == "parallel":
    c = rank
    while c < len(qlist):
        try:
            run(qlist[c])
        except Exception as e:
            print "Error processing task: {} ({})".format(c, qlist[c])
            raise
            # print e
            # raise Exception("Error processing task: {0}".format(qlist[c]))
        c += size
    comm.Barrier()
elif mode == "serial":
    for c in range(len(qlist)):
            run(qlist[c])
else:
    raise ValueError("Invalid `mode` value for script ({}).".format(mode))
