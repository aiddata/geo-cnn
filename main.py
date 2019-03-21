"""

qsub -I -l nodes=1:hima:gpu:ppn=64 -l walltime=8:00:00

hi06 = p100 x2
hi07 = v100 x2



"""


# import os
# import errno
# from affine import Affine

# def make_dir(path):
#     try:
#         os.makedirs(path)
#     except OSError as exception:
#         if exception.errno != errno.EEXIST:
#             raise


from __future__ import print_function, division

import os
import copy
import itertools
import datetime
import time
import pprint

import json
import hashlib

import pandas as pd
import numpy as np
import fiona

import torch
import torch.nn as nn

import resnet

from load_data import build_dataloaders
from data_prep import *
from runscript import *

# -----------------------------------------------------------------------------

quiet = False


batch = False
batch = True

run = {
    "train": True,
    "test": False,
    "predict": False,
    "predict_lsms": True
}

# -----------------------------------------------------------------------------

results = []

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')


def output_csv():
    col_order = [
        "hash",
        "acc",
        "time",
        "run_type",
        "n_epochs",
        "optim",
        "lr",
        "momentum",
        "step_size",
        "gamma",
        "n_input_channels",
        "pixel_size",
        "ncats",
        "loss_weights",
        "train_class_sizes",
        "val_class_sizes",
        "class_acc",
        "net",
        "batch_size",
        "num_workers",
        "dim",
        "agg_method"
    ]
    df_out = pd.DataFrame(results)
    df_out['pixel_size'] = pixel_size
    df_out['ncats'] = ncats
    df_out["train_class_sizes"] = [train_class_sizes] * len(df_out)
    df_out["val_class_sizes"] = [val_class_sizes] * len(df_out)
    df_out = df_out[col_order]
    df_out_path = os.path.join(base_path, "output/results_{}.csv".format(timestamp))
    df_out.to_csv(df_out_path, index=False, encoding='utf-8')


def json_sha1_hash(hash_obj):
    hash_json = json.dumps(hash_obj,
                           sort_keys = True,
                           ensure_ascii = True,
                           separators=(', ', ': '))
    hash_builder = hashlib.sha1()
    hash_builder.update(hash_json)
    hash_sha1 = hash_builder.hexdigest()
    return hash_sha1


# -----------------------------------------------------------------------------


if not batch:

    params = {
        "run_type": 1,
        "n_input_channels": 8,
        "n_epochs": 1,
        "optim": "sgd",
        "lr": 0.009,
        "momentum": 0.95,
        "step_size": 15,
        "gamma": 0.1,
        "loss_weights": [1.0, 1.0, 1.0],
        "net": "resnet152",
        "batch_size": 64,
        "num_workers": 16,
        "dim": 224,
        "agg_method": "min"
    }

    full_param_hash = json_sha1_hash(params)
    param_hash = full_param_hash[:7]

    print("\nParam hash: {}\n".format(param_hash))

    state_path = os.path.join(base_path, "output/s1_state_dict/state_dict_{}.pt".format(param_hash))

    dataloaders = build_dataloaders(
        dataframe_dict,
        base_path,
        data_transform=None,
        dim=params["dim"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        agg_method=params["agg_method"])


    # -----------------

    if run["train"]:

        train_cnn = RunCNN(
            dataloaders, device, cat_names,
            parallel=False, quiet=False, **params)

        acc_p, class_p, time_p = train_cnn.train()

        params['hash'] = param_hash
        params['acc'] = acc_p
        params['class_acc'] = class_p
        params['time'] = time_p
        results.append(params)
        output_csv()

        train_cnn.save(state_path)

    # -----------------

    if run["test"]:

        test_cnn = RunCNN(
            dataloaders, device, cat_names,
            parallel=False, quiet=False, **params)

        test_cnn.load(state_path)

        epoch_loss, epoch_acc, class_acc, time_elapsed = test_cnn.test()

    # -----------------

    if run["predict"]:

        predict_cnn = RunCNN(
            dataloaders, device, cat_names,
            parallel=False, quiet=False, **params)

        predict_cnn.load(state_path)

        pred_data, time_elapsed = predict_cnn.predict(features=True)

    # -----------------

    if run["predict_lsms"]:

        """
        lsms locations dataframe for prediction
        - lat lon
        - consumption
        - ntl (for buffer/box)
        """
        # lsms_cluster['label'] = 1

        lsms_predict = {
            "predict": lsms_cluster.copy(deep=True)
        }

        lsms_dataloaders = build_dataloaders(
            lsms_predict,
            base_path,
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)

        predict_cnn = RunCNN(
            lsms_dataloaders, device, cat_names,
            parallel=False, quiet=False, **params)

        predict_cnn.load(state_path)



        """
        run predict
        512 feats to csv

        append to lsms data for linear regressions
        """
        pred_data, time_elapsed = predict_cnn.predict(features=True)

        feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]

        pred_dicts = [dict(zip(feat_labels, i)) for i in pred_data]
        pred_df = pd.DataFrame(pred_dicts)

        lsms_out = lsms_predict["predict"].merge(pred_df, left_index=True, right_index=True)

        col_order = list(lsms_predict["predict"].columns) + feat_labels
        lsms_out = lsms_out[col_order]

        lsms_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}.csv".format(param_hash, timestamp))

        lsms_out.to_csv(lsms_out_path, index=False, encoding='utf-8')



# -----------------------------------------------------------------------------



if batch:

    # pranges = {
    #     "run_type": [1,2],
    #     "n_input_channels": [8],
    #     "n_epochs": [60],
    #     "lr": [ 0.0005, 0.001, 0.005, 0.01, 0.05],
    #     "momentum": [0.5, 0.7, 0.9, 1.1, 1.3],
    #     "step_size": [5, 10, 15],
    #     "gamma": [0.01, 0.05]
    # }

    pranges = {
        "run_type": [1],
        "n_input_channels": [8],
        "n_epochs": [30],
        "optim": ["sgd"],
        "lr": [0.005, 0.010, 0.015],
        "momentum": [0.95],
        "step_size": [5],
        "gamma": [0.1, 0.001],
        "loss_weights": [
            # [0.1, 0.4, 1.0],
            # [0.4, 0.4, 1.0],
            # [0.8, 0.4, 1.0]
            [1.0, 1.0, 1.0]
        ],
        "net": ["resnet101"],
        "batch_size": [64],
        "num_workers": [16],
        "dim": [224],
        "agg_method": ["mean", "max", "min"]
    }

    print("\nPreparing following parameter set:\n")
    pprint.pprint(pranges, indent=4)
    print('-' * 20)

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    pcount = np.prod([len(i) for i in pranges.values()])

    for ix, p in enumerate(dict_product(pranges)):

        params = copy.deepcopy(p)

        print('-' * 10)
        print("\nParameter combination: {}/{}".format(ix+1, pcount))

        full_param_hash = json_sha1_hash(params)
        param_hash = full_param_hash[:7]

        print("\nParam hash: {}\n".format(param_hash))

        state_path = os.path.join(base_path, "output/s1_state_dict/state_dict_{}.pt".format(param_hash))

        dataloaders = build_dataloaders(
            dataframe_dict,
            base_path,
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"])

        # -----------------

        if run["train"]:

            train_cnn = RunCNN(
                dataloaders, device, cat_names,
                parallel=False, quiet=False, **params)

            acc_p, class_p, time_p = train_cnn.train()

            params['hash'] = param_hash
            params['acc'] = acc_p
            params['class_acc'] = class_p
            params['time'] = time_p
            results.append(params)
            output_csv()

            train_cnn.save(state_path)

        # -----------------

        if run["test"]:

            test_cnn = RunCNN(
                dataloaders, device, cat_names,
                parallel=False, quiet=False, **params)

            test_cnn.load(state_path)

            epoch_loss, epoch_acc, class_acc, time_elapsed = test_cnn.test()

        # -----------------

        if run["predict"]:

            predict_cnn = RunCNN(
                dataloaders, device, cat_names,
                parallel=False, quiet=False, **params)

            predict_cnn.load(state_path)

            pred_data, time_elapsed = predict_cnn.predict(features=True)

        # -----------------

        if run["predict_lsms"]:

            """
            lsms locations dataframe for prediction
            - lat lon
            - consumption
            - ntl (for buffer/box)
            """
            lsms_predict = {
                "predict": lsms_cluster.copy(deep=True)
            }

            lsms_dataloaders = build_dataloaders(
                lsms_predict,
                base_path,
                data_transform=None,
                dim=params["dim"],
                batch_size=params["batch_size"],
                num_workers=params["num_workers"],
                agg_method=params["agg_method"],
                shuffle=False)

            predict_cnn = RunCNN(
                lsms_dataloaders, device, cat_names,
                parallel=False, quiet=False, **params)

            predict_cnn.load(state_path)


            """
            run predict
            512 feats to csv

            append to lsms data for linear regressions
            """
            pred_data, time_elapsed = predict_cnn.predict(features=True)

            feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]

            pred_dicts = [dict(zip(feat_labels, i)) for i in pred_data]
            pred_df = pd.DataFrame(pred_dicts)

            lsms_out = lsms_predict["predict"].merge(pred_df, left_index=True, right_index=True)

            col_order = list(lsms_predict["predict"].columns) + feat_labels
            lsms_out = lsms_out[col_order]

            lsms_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}.csv".format(param_hash, timestamp))

            lsms_out.to_csv(lsms_out_path, index=False, encoding='utf-8')
