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

import pandas as pd
import numpy as np
import fiona

import torch
import torch.nn as nn

import resnet

from load_data import build_dataloaders
from data_prep import *
from runscript import *


quiet = False

results = []

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

df_output_path = os.path.join(
    base_path,
    "run_data/results_{}.csv".format(timestamp))


def output_csv():
    col_order = [
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
    df_out.to_csv(df_output_path, index=False, encoding='utf-8')


# batch = True
batch = False



# -----------------------------------------------------------------------------



# if not batch:

params = {
    "run_type": 2,
    "n_input_channels": 8,
    "n_epochs": 2,
    "optim": "sgd",
    "lr": 0.009,
    "momentum": 0.95,
    "step_size": 15,
    "gamma": 0.1,
    "loss_weights": [1.0, 1.0, 1.0],
    "net": "resnet18",
    "batch_size": 150,
    "num_workers": 16,
    "dim": 224,
    "agg_method": "mean"
}

dataloaders = build_dataloaders(
    dataframe_dict,
    base_path,
    data_transform=None,
    dim=params["dim"],
    batch_size=params["batch_size"],
    num_workers=params["num_workers"],
    agg_method=params["agg_method"])


state_dict_path = os.path.join(base_path, "saved_state_dict.pt")

# -----------------

train_cnn = RunCNN(
    dataloaders, device, cat_names,
    parallel=False, quiet=False, **params)

acc_p, class_p, time_p = train_cnn.train()

params['acc'] = acc_p
params['class_acc'] = class_p
params['time'] = time_p
results.append(params)
output_csv()

train_cnn.save(state_dict_path)

# -----------------

test_cnn = RunCNN(
    dataloaders, device, cat_names,
    parallel=False, quiet=False, **params)

test_cnn.load(state_dict_path)

epoch_loss, epoch_acc, class_acc, time_elapsed = test_cnn.test()

# -----------------

predict_cnn = RunCNN(
    dataloaders, device, cat_names,
    parallel=False, quiet=False, **params)

predict_cnn.load(state_dict_path)

full_preds, time_elapsed = predict_cnn.predict(features=False)
full_preds_2, time_elapsed_2 = predict_cnn.predict(features=True)

# -----------------





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
        "run_type": [2],
        "n_input_channels": [8],
        "n_epochs": [30],
        "optim": ["sgd"],
        "lr": [0.008, 0.009, 0.010],
        "momentum": [0.95],
        "step_size": [15, 30],
        "gamma": [0.1, 0.01],
        "loss_weights": [
            # [0.1, 0.4, 1.0],
            # [0.4, 0.4, 1.0],
            # [0.8, 0.4, 1.0]
            [1.0, 1.0, 1.0]
        ],
        "net": ["resnet50", "resnet152"],
        "batch_size": [128],
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
        print('-' * 10)
        print("\nParameter combination: {}/{}".format(ix+1, pcount))

        dataloaders = build_dataloaders(
            dataframe_dict,
            base_path,
            data_transform=None,
            dim=p["dim"],
            batch_size=p["batch_size"],
            num_workers=p["num_workers"],
            agg_method=p["agg_method"])

        model_p, acc_p, class_p, time_p = run(
            dataloaders, device, mode="train", quiet=quiet, **p)

        pout = copy.deepcopy(p)
        pout['acc'] = acc_p
        pout['class_acc'] = class_p
        pout['time'] = time_p
        results.append(pout)
        output_csv()
