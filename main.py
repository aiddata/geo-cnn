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
from load_survey_data import surveys


# -----------------------------------------------------------------------------

quiet = False


cuda_device_id = 0

device = torch.device("cuda:{}".format(cuda_device_id) if torch.cuda.is_available() else "cpu")

print("Running on:", device)

# run_types = {
#     1: 'fine tuning',
#     2: 'fixed feature extractor'
# }


# mode = "hash"
mode = "batch"
mode = "other"

run = {
    "train": True,
    "test": False,
    "predict": False,
    "predict_new": True
}

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

date_str = datetime.datetime.now().strftime("%Y%m%d")

# -----------------

# training settings
imagery_year = 2010

tags = [date_str, "2010", "true_raw_ntl"]

# -----------------

# prediciton settings
new_predict_source_data = surveys["dhs2010_cluster"].copy(deep=True)
# new_predict_source_data = surveys["lsms2010_cluster"].copy(deep=True)
# new_predict_source_data = surveys["lsms2012_cluster"].copy(deep=True)

pred_tags = [date_str, "2010", "true_raw_ntl"]


# -----------------------------------------------------------------------------


def output_csv():
    col_order = [
        "hash",
        "id_string",
        "tags",
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
    df_out['ncats'] = len(cat_names)
    df_out["train_class_sizes"] = [train_class_sizes] * len(df_out)
    df_out["val_class_sizes"] = [val_class_sizes] * len(df_out)
    df_out = df_out[col_order]
    df_out_path = os.path.join(base_path, "output/s1_train/train_{}.csv".format(timestamp))
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

if mode == "hash":

    # hash_list = []
    hash_list = pd.read_csv(os.path.join(base_path, "_old/cnn_results_merge.csv"))["hash"]

    param_dicts = []

    for param_hash in hash_list:
        param_json_path = os.path.join(base_path, "output/s1_params/params{}.json".format(param_hash))
        with open(param_json_path) as j:
            param_dicts.append(json.load(j))


elif mode == "batch":

    pranges = {
        "tags": [tags],
        "run_type": [1],
        "n_input_channels": [8],
        "n_epochs": [60],
        "optim": ["sgd", "adam"],
        "lr": [0.001, 0.005, 0.01],
        "momentum": [0.97],
        "step_size": [5, 15],
        "gamma": [0.75, 0.25],
        "loss_weights": [
            [1.0, 1.0, 1.0]
        ],
        "net": ["resnet18"],
        # "net": ["resnet101"],
        "batch_size": [64],
        "num_workers": [16],
        "dim": [224],
        "agg_method": ["mean"]
        # "agg_method": ["max", "min"]
        # "agg_method": ["mean", "max", "min"]
    }

    print("\nPreparing following parameter set:\n")
    pprint.pprint(pranges, indent=4)
    print('-' * 20)

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    param_dicts = dict_product(pranges)

else:

    param_dicts = [{
        "tags": tags,
        "run_type": 1,
        "n_input_channels": 8,
        "n_epochs": 5,
        "optim": "adam",
        "lr": 0.005,
        "momentum": 0.97,
        "step_size": 5,
        "gamma": 0.5,
        "loss_weights": [1.0, 1.0, 1.0],
        "net": "resnet18",
        "batch_size": 64,
        "num_workers": 16,
        "dim": 224,
        "agg_method": "mean"
    }]


if mode == "batch":
    pcount = np.prod([len(i) for i in pranges.values()])
else:
    pcount = len(param_dicts)


# -----------------------------------------------------------------------------


results = []

for ix, p in enumerate(param_dicts):

    params = copy.deepcopy(p)

    print('-' * 10)
    print("\nParameter combination: {}/{}".format(ix+1, pcount))

    full_param_hash = json_sha1_hash(params)
    param_hash = full_param_hash[:7]

    print("\nParam hash: {}\n".format(param_hash))

    param_json_path = os.path.join(base_path, "output/s1_params/params_{}.json".format(param_hash))

    with open(param_json_path, "w", 0) as param_json:
        json.dump(params, param_json)

    state_path = os.path.join(base_path, "output/s1_state/state_{}.pt".format(param_hash))

    # -----------------

    if run["train"] or run["test"] or run["predict"]:

        dataloaders = build_dataloaders(
            dataframe_dict,
            base_path,
            imagery_year,
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"])

        train_cnn = RunCNN(
            dataloaders, device, cat_names,
            parallel=False, quiet=quiet, **params)

        if run["train"]:
            acc_p, class_p, time_p = train_cnn.train()

            params['hash'] = param_hash
            params['id_string'] = "{}_{}.csv".format(param_hash, timestamp)
            params['acc'] = acc_p
            params['class_acc'] = class_p
            params['time'] = time_p
            results.append(params)
            output_csv()

            train_cnn.save(state_path)
        else:
            train_cnn.load(state_path)


        if run["test"]:
            epoch_loss, epoch_acc, class_acc, time_elapsed = train_cnn.test()

        if run["predict"]:
            pred_data, _ = train_cnn.predict(features=True)

    # -----------------

    if run["predict_new"]:

        """
        - load data
        - load trained cnn state
        - run predict
        - append cnn features to original data
        - output to csv for second stage models
        """

        new_data = {
            "predict": new_predict_source_data
        }

        new_dataloaders = build_dataloaders(
            new_data,
            base_path,
            imagery_year,
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)

        new_cnn = RunCNN(
            new_dataloaders, device, cat_names,
            parallel=False, quiet=quiet, **params)

        new_cnn.load(state_path)

        # ---------

        new_pred_data, _ = new_cnn.predict(features=True)

        feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]

        pred_dicts = [dict(zip(feat_labels, i)) for i in new_pred_data]
        pred_df = pd.DataFrame(pred_dicts)

        new_out = new_data["predict"].merge(pred_df, left_index=True, right_index=True)

        new_out["pred_tags"] = [pred_tags] * len(new_out)

        col_order = ["pred_tags"] + list(new_data["predict"].columns) + feat_labels
        new_out = new_out[col_order]

        new_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}.csv".format(param_hash, timestamp))

        new_out.to_csv(new_out_path, index=False, encoding='utf-8')
