

import os
import itertools

import numpy as np
import pandas as pd

from sklearn import linear_model, metrics
# from sklearn.model_selection import train_test_split

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

# def quick(x,y, model):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)
#     model.fit(x_train, y_train)
#     model_preds = model.predict(x_test)
#     r2 = r2_score(y_test, model_preds)
#     return r2


# print "NTL (Linear)"
# lm_ntl = quick(lsms_out[['ntl_2010']], lsms_out["cons"], model=LinearRegression())
# print lm_ntl

# print "Features (Linear)"
# lm_feat = quick(lsms_out[test_feat_labels], lsms_out["cons"], model=LinearRegression())
# print lm_feat

# print "NTL (Ridge)"
# rr_ntl = quick(lsms_out[['ntl_2010']], lsms_out["cons"], model=Ridge())
# print rr_ntl

# print "Features (Ridge)"
# rr_feat = quick(lsms_out[test_feat_labels], lsms_out["cons"], model=Ridge())
# print rr_feat

# -------------------------------------


x_train_ntl = lsms_out[['ntl_2010']]
x_train_feat = lsms_out[test_feat_labels]
y_train = lsms_out["cons"]

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
