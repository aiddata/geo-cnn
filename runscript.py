"""

qsub -I -l nodes=1:hima:gpu:ppn=64 -l walltime=8:00:00


mpirun -mca orte_base_help_aggregate 0 --mca mpi_warn_on_fork 0 --map-by node -np 32 python-mpi lsms_imagery_prep.py


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
import glob
import itertools
import datetime
import time
import pprint

import rasterio
import pandas as pd
import numpy as np
import fiona


import torchvision
from torchvision import utils, datasets, models, transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import resnet

from load_data import BandDataset
from create_grid import PointGrid

from data_prep import *


# -----------------------------------------------------------------------------



def build_dataloaders(data_transform=None, dim=224, batch_size=64, num_workers=16):

    if data_transform == None:

        data_transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # imagenet means
                                 std=[0.229, 0.224, 0.225]), # imagenet stds
        ])

    train_dset = BandDataset(train_df, base_path, dim=dim, transform=data_transform)
    val_dset = BandDataset(val_df, base_path, dim=dim, transform=data_transform)
    test_dset = BandDataset(test_df, base_path, dim=dim, transform=data_transform)

    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }

    return dataloaders


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, quiet=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if not quiet:
            print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_correct = 0
            running_count = 0

            class_correct = [0] * len(cat_names)
            class_count = [0] * len(cat_names)

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_correct += torch.sum(preds == labels.data)
                running_count += inputs.size(0)

                for i in range(len(cat_names)):
                    label_indexes = (labels == i).nonzero().squeeze()
                    class_correct[i] += torch.sum(preds[label_indexes] == labels[label_indexes]).item()
                    class_count[i] += len(label_indexes)


            epoch_loss = running_loss / running_count
            epoch_acc = running_correct.item() / running_count

            class_acc = [class_correct[i] / class_count[i] for i in range(len(cat_names))]

            if not quiet:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_class_acc = class_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                for i in range(len(cat_names)):
                    print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
                        i, class_correct[i], class_count[i], class_acc[i]))


    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_class_acc, time_elapsed


def test_model(model, criterion, optimizer):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_correct = 0
    running_count = 0

    class_correct = [0] * len(cat_names)
    class_count = [0] * len(cat_names)

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)

        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(0):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)

        running_correct += torch.sum(preds == labels.data)
        running_count += inputs.size(0)

        for i in range(len(cat_names)):
            label_indexes = (labels == i).nonzero().squeeze()
            class_correct[i] += sum(preds[label_indexes] == labels[label_indexes]).item()
            class_count[i] += len(label_indexes)


    epoch_loss = running_loss / running_count
    epoch_acc = running_correct.item() / running_count

    class_acc = [class_correct[i] / class_count[i] for i in range(len(cat_names))]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

    for i in range(len(cat_names)):
        print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
            i, class_correct[i], class_count[i], class_acc[i]))


    time_elapsed = time.time() - since
    print('\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return epoch_loss, epoch_acc, class_acc, time_elapsed



def predict_model(model):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    full_preds = []

    # iterate over data
    for inputs, _ in dataloaders[phase]:
        inputs = inputs.to(device)

        with torch.set_grad_enabled(0):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            full_preds += preds # need to test this

    time_elapsed = time.time() - since
    print('\nPrediction completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return full_preds, time_elapsed




def run(quiet=False, **kwargs):

    print("\n{}:\n".format(run_types[kwargs["run_type"]]))
    print(kwargs)

    if kwargs["net"] == "resnet18":
        model_x = resnet.resnet18(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    elif kwargs["net"] == "resnet34":
        model_x = resnet.resnet34(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    elif kwargs["net"] == "resnet50":
        model_x = resnet.resnet50(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    elif kwargs["net"] == "resnet101":
        model_x = resnet.resnet101(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    elif kwargs["net"] == "resnet152":
        model_x = resnet.resnet152(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    else:
        raise Exception("net not found")

    if kwargs["run_type"] == 2:
        for param in model_x.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_x.fc.in_features

    model_x.fc = nn.Linear(num_ftrs, ncats)


    model_x = model_x.to(device)

    loss_weights = torch.tensor(
        map(float, kwargs["loss_weights"])).cuda()

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    if kwargs["optim"] == "sgd":
        # Observe that only parameters of final layer are being optimized as opposed to before.
        optimizer_x = optim.SGD(model_x.fc.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"])

    # Decay LR by a factor of `gamma` every `step_size` epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_x, step_size=kwargs["step_size"], gamma=kwargs["gamma"])


    model_x, acc_x, class_x, time_x = train_model(
        model_x, criterion, optimizer_x, exp_lr_scheduler,
        num_epochs=kwargs["n_epochs"], quiet=quiet)

    return model_x, acc_x, class_x, time_x





if __name__ == "__main__":

    quiet = False

    results = []

    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y_%m_%d_%H_%M_%S')
    df_output_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/run_data/results_{}.csv".format(timestamp)

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
            "dim"
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
            "net": "resnet18",
            "batch_size": 64,
            "num_workers": 16,
            "dim": 300
        }

        dataloaders = build_dataloaders(data_transforms=None, dim=params["dim"], batch_size=params["batch_size"], num_workers=params["num_workers"])
        model_p, acc_p, class_p, time_p = run(quiet=quiet, **params)
        params['acc'] = acc_p
        params['class_acc'] = class_p
        params['time'] = time_p
        results.append(params)
        output_csv()


    # -------------------------------------

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
            "n_epochs": [10],
            "optim": ["sgd"],
            "lr": [0.009],
            "momentum": [0.95],
            "step_size": [15],
            "gamma": [0.1],
            "loss_weights": [
                # [0.1, 0.4, 1.0],
                # [0.4, 0.4, 1.0],
                # [0.8, 0.4, 1.0]
                [1.0, 1.0, 1.0]
            ],
            "net": ["resnet18"],
            "batch_size": [64],
            "num_workers": [16],
            "dim": [224, 300, 400]
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
            dataloaders = build_dataloaders(data_transforms=None, dim=p["dim"], batch_size=p["batch_size"], num_workers=p["num_workers"])
            model_p, acc_p, class_p, time_p = run(quiet=quiet, **p)
            pout = copy.deepcopy(p)
            pout['acc'] = acc_p
            pout['class_acc'] = class_p
            pout['time'] = time_p
            results.append(pout)
            output_csv()
