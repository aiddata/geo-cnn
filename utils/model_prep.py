
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn import linear_model, metrics, neural_network
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from scipy import stats

import matplotlib.pyplot as plot

import utils.load_custom_covar


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


def run_cv(X, y, model, k, k_inner=5, alphas=None, metric=None, randomize=False, **kwargs):
    """
    Runs nested cross-validation to make predictions and compute r-squared.

    k_inner, alphas, metric only needed when determining ideal alpha param within folds

    (could add repeats to this, similar to demo cross_validate function above)

    """
    if "params" in kwargs:
        params = kwargs["params"]
    else:
        params = {}
    best_alpha = None
    best_alpha_list = []
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
            best_alpha = find_best_alpha(X_train, y_train, k_inner, model, metric, alphas, params)
            best_alpha_list.append(best_alpha)
            params["alpha"] = best_alpha
        X_train, X_test = scale_features(X_train, X_test)
        y_test_predict = train_and_predict(X_train, y_train, X_test, model, params)
        y_true.append(y_test)
        y_predict.append(y_test_predict)
        # score[fold] = metric(y_test, y_test_predict)
    # return score.mean(), y_true, y_predict
    final_alpha = np.mean(best_alpha_list) if best_alpha_list else best_alpha
    return y_true, y_predict, final_alpha


def find_best_alpha(X, y, k_inner, model, metric, alphas, params):
    """
    Finds the best alpha in an inner CV loop.
    """
    kfa = KFold(n_splits=k_inner, shuffle=True)
    best_alpha = 0
    best_score = 0
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kfa.split(y):
            params["alpha"] = alpha
            y_hat = predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, model, params)
        score = metric(y, y_hat)
        if score > best_score:
            best_alpha = alpha
            best_score = score
    return best_alpha


def predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, model, params):
    """
    Predicts inner test fold.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)
    y_hat[test_idx] = train_and_predict(X_train, y_train, X_test, model, params)
    return y_hat


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    X_scaler = StandardScaler(with_mean=True, with_std=False)
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test


def train(X_train, y_train, model, params):
    """
    Trains model and predicts test set.
    """
    lm = model(**params)
    lm.fit(X_train, y_train)
    return lm


def train_and_predict(X_train, y_train, X_test, model, params):
    """
    Trains model and predicts test set.
    """
    lm = train(X_train, y_train, model, params)
    y_predict = lm.predict(X_test)
    return y_predict


def run_models(id_string, model_helper):
    """
    id_string = <train hash>_<predict hash>_<version tag>_<predict tag>
    """
    print "S2Running: {}".format(id_string)
    mh = model_helper

    model_tag = mh.settings.config["model_tag"]
    base_path = mh.settings.base_path

    pred_data_path = os.path.join(base_path, "output/s1_predict/predict_{}.csv".format(id_string))

    pred_data = pd.read_csv(pred_data_path, quotechar='\"',
                        na_values='', keep_default_na=False,
                        encoding='utf-8')

    feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]



    input_list = mh.settings.data["second_stage"]["inputs"]

    # reduce feature dimensions using PCA
    pca_dimension = 15
    pca = PCA(n_components=pca_dimension)

    y_train = pred_data["pred_yval"].values

    # add all standard x_train options
    x_train = {}
    x_train["ntl"] = pred_data[['ntl']].values
    x_train["cnn"] = pred_data[feat_labels].values

    x_train["all"] = pred_data[feat_labels+["ntl"]].values
    x_train["all-pca{}".format(pca_dimension)] = pca.fit_transform(x_train["all"])

    x_train["cnn-pca{}".format(pca_dimension)] = pca.fit_transform(x_train["cnn"])
    x_train["cnn-pca{}-ntl".format(pca_dimension)] = np.append(x_train["cnn-pca{}".format(pca_dimension)], x_train["ntl"], 1)

    # add custom x_train options
    for i in mh.settings.data["second_stage"]["custom_definitions"]:
        cfunc = getattr(load_custom_covar, i["function"])
        new_var = cfunc(pred_data[["lon", "lat"]])
        x_train[i["name"]] = new_var
        for j in i["inputs"]:
            cname = "{}-{}".format(i["name"], j)
            x_train[cname] = np.append(x_train[j], x_train[i["name"]], 1)
            input_list.append(cname)

    # delete standard x_train options not specified by user
    for i in x_train.keys():
        if i not in input_list:
            del x_train[i]


    print("Running models:")

    for name in mh.model_list:

        lm_dict = deepcopy(mh.model_lookup[name])

        lm_params = lm_dict["params"] if "params" in lm_dict else {}

        results = []

        metrics_results_path = os.path.join(base_path, "output/s2_metrics/metrics_{}_{}_{}.csv".format(name, id_string, model_tag))



        for x_name, x_data in x_train.iteritems():

            print "\t{}({})...".format(name, x_name)

            models_results_path = os.path.join(base_path, "output/s2_models/models_{}_{}_{}_{}.joblib".format(name, x_name, id_string, model_tag))

            # run with or without cross validation
            if "k" in lm_dict:

                try:
                    y_true, y_predict, best_alpha = run_cv(x_data, y_train, **lm_dict)

                    if best_alpha is not None and "alpha" in lm_params:
                        if best_alpha != lm_params["alpha"]:
                            warnings.warn("\tBest alpha does not match specified alpha")
                        lm_parmas["alpha"] = best_alpha

                    lm = train(x_data, y_train, lm_dict["model"], params=lm_params)
                    joblib.dump(lm, models_results_path)

                except Exception as e:
                    print(e)
                    metric_vals = ["Error" for i in mh.metric_list]
                else:
                    # print y_true
                    # print "========="
                    # print y_predict
                    # raise
                    metric_vals = [
                        np.array( [mh.metric_list[j](y_true[i], y_predict[i]) for i in range(lm_dict["k"])] ).mean()
                        for j in mh.metric_list
                    ]
            else:

                if "alphas" in lm_dict:
                    raise Exception("Must specify alpha manually if not using cross-validation")

                lm = train(x_data, y_train, lm_dict["model"], params=lm_params)

                try:
                    y_predict = lm.predict(x_data, **lm_params)
                except Exception as e:
                    print(e)
                    metric_vals = ["Error" for i in mh.metric_list]
                else:
                    metric_vals = [mh.metric_list[i](y_train, y_predict) for i in mh.metric_list]

            tmp_vals = [id_string, name, lm_dict["model"].__name__, x_name]
            tmp_vals += metric_vals
            results.append(dict(zip(mh.keys, tmp_vals)))

        df = pd.DataFrame(results)
        df = df[mh.keys]
        df.to_csv(metrics_results_path, index=False, encoding='utf-8')



def run_tasks(tasks, func, args, mode="auto", raise_errors=True):
    parallel = False
    if mode in ["auto", "parallel"]:
        try:
            from mpi4py import MPI
            parallel = True
        except:
            parallel = False
    elif mode != "serial":
        raise Exception("Invalid `mode` value for script.")
    if parallel:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 1
        rank = 0
    c = rank
    while c < len(tasks):
        try:
            func(tasks[c], args)
        except Exception as e:
            print "Error processing: {0}".format(tasks[c])
            if raise_errors:
                raise
            else:
                print e
        c += size
    if parallel:
        comm.Barrier()


# -------------------------------------


def pearson_r2(true, predict):
    return stats.pearsonr(true, predict)[0] ** 2


class ModelHelper():

    def __init__(self, settings):

        self.settings = settings

        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        self.model_lookup = {
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
                "alphas": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
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
            },
            "logistic-cv10": {
                "model": linear_model.LogisticRegression,
                "k": 10
            },
            "ridgeclassifier-cv10": {
                "model": linear_model.RidgeClassifier,
                "k": 10,
                "k_inner": 10,
                "alphas": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
                "metric": pearson_r2
            },
            "mlpclassifier-cv10": {
                "model": neural_network.MLPClassifier,
                "k": 10,
                "params": {
                    "hidden_layer_sizes": (512, ),
                    "max_iter": 2000
                }
            }
        }


        # https://scikit-learn.org/stable/modules/classes.html#regression-metrics
        self.metric_lookup = {
            "pr2": pearson_r2,
            "evs": metrics.explained_variance_score,
            "mae": metrics.mean_absolute_error,
            "mae2": metrics.median_absolute_error,
            "mse": metrics.mean_squared_error,
            "r2": metrics.r2_score,
            "recall": metrics.recall_score,
            "precision": metrics.precision_score,
            "f1": metrics.f1_score,
            "accuracy": metrics.accuracy_score
        }

        self.model_list = self.settings.data["second_stage"]["models"]

        self.metric_list = {i: self.metric_lookup[i] for i in self.settings.data["second_stage"]["metrics"]}

        self.keys = ["id", "name", "model", "input"] + self.metric_list.keys()
