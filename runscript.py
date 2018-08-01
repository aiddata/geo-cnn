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


print("Initializing...")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running on:", device)



# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

ncats = len(cat_names)


# -----------------------------------------------------------------------------


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

lsms_clusters_path = os.path.join(
    base_path, "survey_data/final/tanzania_lsms_cluster.csv")

lsms_cluster = pd.read_csv(lsms_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')


consumption_list = list(lsms_cluster['cons'])
cat_vals = [np.percentile(consumption_list, x*100/ncats) for x in range(1, ncats+1)]


def classify(val):
    for cix, cval in enumerate(cat_vals):
        if val <= cval:
            return cat_names[cix]
    print(val)
    raise Exception("Could not classify")

lsms_cluster['label'] = lsms_cluster.apply(
    lambda z: classify(z['cons']), axis=1)


# -----------------------------------------------------------------------------


ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
ntl_year = 2010
ntl_path = glob.glob(os.path.join(ntl_base, "*{0}*.tif".format(ntl_year)))[0]
ntl_file = rasterio.open(ntl_path)

def get_ntl(lon, lat, ntl_dim=7):
    """Get nighttime lights average value for grid around point
    """
    r, c = ntl_file.index(lon, lat)
    ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
    ntl_data = ntl_file.read(1, window=ntl_win)
    ntl_mean = ntl_data.mean()
    return ntl_mean


lsms_cluster['ntl'] = lsms_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)


class_ntl_means = dict(zip(cat_names, lsms_cluster.groupby('label')['ntl'].mean()))


# -----------------------------------------------------------------------------

print("Preparing grid")

# boundary
tza_adm0_path = os.path.join(base_path, 'TZA_ADM0_GADM28_simplified.geojson')
tza_adm0 = fiona.open(tza_adm0_path)


# define, build, and save sample grid
pixel_size = 0.008

grid = PointGrid(tza_adm0)
grid.grid(pixel_size)

geo_path = os.path.join(base_path, "sample_grid.geojson")
grid.to_geojson(geo_path)

csv_path = os.path.join(base_path, "sample_grid.csv")
grid.to_csv(csv_path)

df = grid.to_dataframe()

# look up ntl values for each grid cell
df['ntl'] = df.apply(lambda z: get_ntl(z['lon'], z['lat'], ntl_dim=7), axis=1)

# find nearest neighbor for each grid cell using class_ntl_means

# set all where ntl <  lowest class automatically to lowest class
df['label'] = None
df.loc[df['ntl'] <= class_ntl_means[min(cat_names)], 'label'] = min(cat_names)


def find_nn(val):
    nn_options = [(k, abs(val - v)) for k, v in class_ntl_means.iteritems()]
    nn_class = sorted(nn_options, key=lambda x: x[1])[0][0]
    return nn_class

# run nearest neigbor class label for remaining
to_label = df['ntl'] > class_ntl_means[min(cat_names)]
df.loc[to_label, 'label'] = df.loc[to_label].apply(lambda z: find_nn(z['ntl']), axis=1)



# drop out some of the extra from low category
original_df = df.copy(deep=True)

# for debugging
# df = original_df.copy(deep=True)



print("Samples per cat (raw):")
for i in cat_names: print("{0}: {1}".format(i, sum(df['label'] == i)))


# ==============================


# low_count = sum(df['label'] == 0)
# other_count = sum(df['label'] == 1) + sum(df['label'] == 2)

# keep_ratio = (other_count * 1.0) / low_count

# keep_ratio = keep_ratio if keep_ratio < 1 else 1

# df['drop'] = 'keep'
# df.loc[df['label'] == 0, 'drop'] = np.random.choice(["drop", "keep"], size=(low_count,), p=[1-keep_ratio, keep_ratio])


# ==============================
# equal class sizes based on smallest class size

raw_class_sizes = [sum(df['label'] == i) for i in cat_names]

nkeep = min(raw_class_sizes)

df['drop'] = 'drop'

for i in cat_names:
    class_index = df.loc[df['label'] == i].index
    n_keep = min(raw_class_sizes)
    keepers = np.random.permutation(class_index)[:nkeep]
    df.loc[keepers, 'drop'] = 'keep'


# ==============================


df = df.loc[df['drop'] == 'keep'].copy(deep=True)

print("Samples per cat (reduced):")
for i in cat_names: print("{0}: {1}".format(i, sum(df['label'] == i)))



# -----------------------------------------------------------------------------


print("Building datasets")


# ==========
# lsms_cluster['type'] = np.random.choice(["train", "val"], size=(len(lsms_cluster),), p=[0.90, 0.10])

# train_df = lsms_cluster.loc[lsms_cluster['type'] == "train"]
# val_df = lsms_cluster.loc[lsms_cluster['type'] == "val"]
# ==========


type_names = ["train", "val", "test"]

# ratio for train, val, test data
# must sum to 1.0
type_weights = [0.85, 0.149, 0.001]

type_sizes = np.zeros(ncats).astype(int)
for i in cat_names:
    type_sizes[i] = int(np.floor(nkeep * type_weights[i]))

# used floor for all counts, so total will always be short
for _ in range(nkeep-sum(type_sizes)):
    type_sizes[np.random.randint(0, ncats)] += 1

type_list = [[type_names[i]] * type_sizes[i] for i in cat_names]
type_list = list(itertools.chain.from_iterable(type_list))

df['type'] = None
for i in cat_names:
    np.random.shuffle(type_list)
    df.loc[df['label'] == i, 'type'] = type_list


train_df = df.loc[df['type'] == "train"]
val_df = df.loc[df['type'] == "val"]
test_df = df.loc[df['type'] == "test"]


train_class_sizes = [sum(train_df['label'] == i) for i in cat_names]
val_class_sizes = [sum(val_df['label'] == i) for i in cat_names]
test_class_sizes = [sum(test_df['label'] == i) for i in cat_names]


print("Samples per cat (train):")
for i in cat_names: print("{0}: {1}".format(i, sum(train_df['label'] == i)))

print("Samples per cat (val):")
for i in cat_names: print("{0}: {1}".format(i, sum(val_df['label'] == i)))

print("Samples per cat (test):")
for i in cat_names: print("{0}: {1}".format(i, sum(test_df['label'] == i)))



# -----------------------------------------------------------------------------


# bands = 8

# imagenet_means = [0.485, 0.456, 0.406]
# imagenet_stds = [0.229, 0.224, 0.225]

# new_means = imagenet_means #+ [np.mean(imagenet_means)] * (bands-3)
# new_stds = imagenet_stds #+ [np.mean(imagenet_stds)] * (bands-3)

data_transform = transforms.Compose([
    # transforms.RandomSizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 128
num_workers = 16

train_dset = BandDataset(train_df, base_path, transform=data_transform)
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


val_dset = BandDataset(val_df, base_path, transform=data_transform)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


test_dset = BandDataset(test_df, base_path, transform=data_transform)
test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



dataloaders = {
    "train": train_dataloader,
    "val": val_dataloader,
    "test": test_dataloader
}



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

            # ==========
            class_correct = [0] * len(cat_names)
            class_count = [0] * len(cat_names)
            # ==========

            # Iterate over data.
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

                # ==========
                for i in range(len(cat_names)):
                    label_indexes = (labels == i).nonzero().squeeze()
                    class_correct[i] += sum(preds[label_indexes] == labels[label_indexes]).item()
                    class_count[i] += len(label_indexes)
                # ==========


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
                # ==========
                for i in range(len(cat_names)):
                    print('Accuracy of class {} : {} / {} = {:.4f} %'.format(
                        i, class_correct[i], class_count[i], class_acc[i]))
                # ==========


    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_class_acc, time_elapsed


def run(quiet=False, **kwargs):

    print("\n{}:\n".format(run_types[kwargs["run_type"]]))
    print(kwargs)

    # model_x = resnet.resnet18(pretrained=True, n_input_channels=kwargs["n_input_channels"])
    model_x = resnet.resnet34(pretrained=True, n_input_channels=kwargs["n_input_channels"])

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


    model_x, acc_x, class_x, time_x = train_model(model_x, criterion, optimizer_x,
                                         exp_lr_scheduler, num_epochs=kwargs["n_epochs"],
                                         quiet=quiet)

    return model_x, acc_x, class_x, time_x


run_types = {
    1: 'fine tuning',
    2: 'fixed feature extractor'
}


if __name__ == "__main__":

    quiet = False

    results = []

    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y_%m_%d_%H_%M_%S')
    df_output_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/results_{}.csv".format(timestamp)

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
            "class_acc"
        ]
        df_out = pd.DataFrame(results)
        df_out['pixel_size'] = pixel_size
        df_out['ncats'] = ncats
        df_out["train_class_sizes"] = [train_class_sizes] * len(df_out)
        df_out["val_class_sizes"] = [val_class_sizes] * len(df_out)
        df_out = df_out[col_order]
        df_out.to_csv(df_output_path, index=False, encoding='utf-8')


    batch = True

    if not batch:

        params = {
            "run_type": 1,
            "n_input_channels": 8,
            "n_epochs": 30,
            "optim": "sgd",
            "lr": 0.0005,
            "momentum": 0.9,
            "step_size": 5,
            "gamma": 0.05,
            "loss_weights": [0.1, 0.4, 1]
        }

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
            "run_type": [1, 2],
            "n_input_channels": [8],
            "n_epochs": [10, 20],
            "optim": ["sgd"],
            "lr": [0.008],
            "momentum": [0.95],
            "step_size": [15],
            "gamma": [0.01],
            "loss_weights": [
                # [0.1, 0.4, 1.0],
                # [0.4, 0.4, 1.0],
                # [0.8, 0.4, 1.0]
                [1.0, 1.0, 1.0]
            ]
        }

        def dict_product(d):
            keys = d.keys()
            for element in itertools.product(*d.values()):
                yield dict(zip(keys, element))

        pcount = np.prod([len(i) for i in pranges.values()])

        for ix, p in enumerate(dict_product(pranges)):
            print('-' * 10)
            print("\nParameter combination: {}/{}".format(ix, pcount))
            model_p, acc_p, class_p, time_p = run(quiet=quiet, **p)
            pout = copy.deepcopy(p)
            pout['acc'] = acc_p
            pout['class_acc'] = class_p
            pout['time'] = time_p
            results.append(pout)
            if ix % 10 == 0:
                output_csv()


        output_csv()
