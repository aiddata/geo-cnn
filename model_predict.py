
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib



def run_models(id_string, settings):
    """
    id_string = <train hash>_<predict hash>_<version tag>_<predict tag>
    """
    print "S2Running: {}".format(id_string)

    model_list = settings.data["second_stage"]["models"]

    model_tag = settings.config["model_tag"]
    base_path = settings.base_path

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

    for name in model_list:

        lm_dict = deepcopy(mh.model_lookup[name])

        results = []

        models_results_path = os.path.join(base_path, "output/s2_models/models_{}_{}_{}.joblib".format(name, id_string, model_tag))
        metrics_results_path = os.path.join(base_path, "output/s2_metrics/metrics_{}_{}_{}.csv".format(name, id_string, model_tag))

        keys = ['a', 'b', 'c']
        
        for x_name, x_data in x_train.iteritems():

            print "\t{}({})...".format(name, x_name)

            lm = train(x_data, y_train, lm_dict["model"])
            joblib.dump(lm, models_results_path)


            # Scales features using StandardScaler.
            X_scaler = StandardScaler(with_mean=True, with_std=False)
            X_data = X_scaler.fit_transform(x_data)
            # Trains model and predicts test set.
            y_predict = lm.predict(X_data)


            tmp_vals = [id_string, name] + metric_vals

            results.append(dict(zip(keys, tmp_vals)))

        df = pd.DataFrame(results)
        df = df[keys]
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
