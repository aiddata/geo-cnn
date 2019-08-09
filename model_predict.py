
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from load_ntl_data import NTL_Reader


# pred_data_path, models_results_path = qlist[0]
# settings = s


def run_models(task, settings):
    """
    id_string = <train hash>_<predict hash>_<version tag>_<predict tag>
    """

    pred_data_path, models_results_path = task

    print "S3-S2 Running: \n\tS1: {}\n\tS2: {}".format(
        os.path.basename(pred_data_path),
        os.path.basename(models_results_path)
    )

    pred_data = pd.read_csv(pred_data_path, quotechar='\"',
                        na_values='', keep_default_na=False,
                        encoding='utf-8')

    s3_info = settings.data["third_stage"]
    s3_predict = s3_info["predict"]

    ntl = NTL_Reader(ntl_type=s3_predict["ntl_type"], calibrated=s3_predict["ntl_calibrated"])
    ntl.set_year(s3_predict["ntl_year"])
    pred_data['ntl'] = pred_data.apply(
        lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=s3_predict["ntl_dim"]), axis=1)


    model_tag = settings.config["model_tag"]
    base_path = settings.base_path

    finfo = os.path.basename(models_results_path).split("_")

    model_name = finfo[1]
    param_hash = finfo[3]
    predict_hash = finfo[4]

    s3_s2_string = "_".join(str(i) for i in [
        model_name,
        param_hash,
        predict_hash,
        s3_info["grid"]["boundary_id"],
        s3_info["predict"]["imagery_year"],
        settings.config["version"],
        settings.config["predict_tag"],
        settings.config["model_tag"]
    ])


    results_path = os.path.join(settings.base_path, "output/s3_s2_predict/predict_{}.csv".format(s3_s2_string))


    # reduce feature dimensions using PCA
    pca_dimension = 15
    pca = PCA(n_components=pca_dimension)

    feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]

    x_train = {}
    x_train["ntl"] = pred_data[['ntl']].values
    x_train["cnn"] = pred_data[feat_labels].values

    x_train["all"] = pred_data[feat_labels+["ntl"]].values
    x_train["all-pca{}".format(pca_dimension)] = pca.fit_transform(x_train["all"])

    x_train["cnn-pca{}".format(pca_dimension)] = pca.fit_transform(x_train["cnn"])
    x_train["cnn-pca{}-ntl".format(pca_dimension)] = np.append(x_train["cnn-pca{}".format(pca_dimension)], x_train["ntl"], 1)


    results = {}
    for x_name, x_data in x_train.iteritems():
        print "\t{}...".format(x_name)
        # load trained model
        lm = joblib.load(models_results_path.replace("INPUT", x_name))
        # scale features using StandardScaler
        X_scaler = StandardScaler(with_mean=True, with_std=False)
        X_data = X_scaler.fit_transform(x_data)
        # predict
        y_predict = lm.predict(X_data)
        results[x_name] = y_predict


    df = pred_data[['cell_id','column', 'lat', 'lon', 'row']].copy(deep=True)
    for y_name, y_data in results.iteritems():
        df[y_name] = y_data

    df.to_csv(results_path, index=False, encoding='utf-8')



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
