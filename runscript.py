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

import time
import os
import copy

import rasterio
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils, datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import resnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running on:", device)


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

lsms_clusters_path = os.path.join(
    base_path, "survey_data/final/tanzania_lsms_cluster.csv")

lsms_cluster = pd.read_csv(lsms_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

ncats = 3
cats = range(1, ncats+1)
cat_vals = map(lambda x: np.percentile(list(lsms_cluster['cons']), x*100/ncats),
               cats)

# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

def classify(val):
    for cix, cval in enumerate(cat_vals):
        if val <= cval:
            return cat_names[cix]
    print(val)
    raise Exception("Could not classify")

lsms_cluster['label'] = lsms_cluster.apply(
    lambda z: classify(z['cons']), axis=1)

lsms_cluster['type'] = np.random.choice(["train", "val"], size=(len(lsms_cluster),), p=[0.90, 0.10])

training_df = lsms_cluster.loc[lsms_cluster['type'] == "train"]
validation_df = lsms_cluster.loc[lsms_cluster['type'] == "val"]



class BandDataset(Dataset):
    """Get the data
    """
    def __init__(self, dataframe, root_dir, transform=None):

        self.dim = 224
        self.year = 2010
        # self.bands = ["b1", "b2", "b3", ]
        self.bands = ["b1", "b2", "b3", "b4", "b5", "b7", "b61", "b62"]
        # self.bands = ["b3", "b4", "b5"]

        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        dim = self.dim

        row = self.dataframe.iloc[idx]

        label = row['label']

        lon = row['lon']
        lat = row['lat']

        feature = np.empty((len(self.bands), dim, dim))

        for bnum, band in enumerate(self.bands):

            season_mosaics_path = os.path.join(
                self.root_dir, "season_mosaics/all",
                "{}_all_{}.tif".format(self.year, band))

            season_mosaics = rasterio.open(season_mosaics_path)

            r, c = season_mosaics.index(lon, lat)
            win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
            data = season_mosaics.read(1, window=win)

            if data.shape != (dim, dim):
                raise Exception("bad feature")

            feature[bnum] = data

        if self.transform:
            feature = np.transpose(feature,(1,2,0))
            feature = self.transform(feature)

        # return torch.from_numpy(feature).float(), label
        return feature.float(), label


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

batch_size = 64
num_workers = 16

train_dset = BandDataset(training_df, base_path, transform=data_transform)
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


val_dset = BandDataset(validation_df, base_path, transform=data_transform)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


dataloaders = {
    "train": train_dataloader,
    "val": val_dataloader
}

dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, quiet=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if not quiet:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_count = 0

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
                # print(loss.item())
                # print(inputs.size(0))
                # print(torch.sum(preds == labels.data))
                # print(inputs.size(0) == len(labels.data))
                # print('++++++++++')
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_count += inputs.size(0)


            # print(dataset_sizes[phase])
            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.double() / running_count

            if not quiet:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if not quiet:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, time_elapsed



run_types = {
    1: 'fine tuning',
    2: 'fixed feature extractor'
}


def run(**kwargs):

    print("\n{}:\n".format(run_types[kwargs["run_type"]]))
    print(kwargs)

    model_x = resnet.resnet18(pretrained=True, n_input_channels=kwargs["n_input_channels"])

    if kwargs["run_type"] == 2:
        for param in model_x.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_x.fc.in_features

    model_x.fc = nn.Linear(num_ftrs, ncats)


    model_x = model_x.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_x = optim.SGD(model_x.fc.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_x, step_size=kwargs["step_size"], gamma=kwargs["gamma"])


    model_x, acc_x, time_x = train_model(model_x, criterion, optimizer_x,
                                         exp_lr_scheduler, num_epochs=kwargs["n_epochs"])

    return model_x, acc_x, time_x



if __name__ == "__main__":

    # params = {
    #     "run_type": 1,
    #     "n_input_channels": 8,
    #     "n_epochs": 50,
    #     "lr": 0.001,
    #     "momentum": 0.9,
    #     "step_size": 5,
    #     "gamma": 0.01
    # }

    # run(**params)


    import itertools
    import pandas as pd
    import datetime


    pranges = {
        "run_type": [1,2],
        "n_input_channels": [8],
        "n_epochs": [60],
        "lr": [ 0.0005, 0.001, 0.005, 0.01, 0.05],
        "momentum": [0.5, 0.7, 0.9, 1.1, 1.3],
        "step_size": [5, 10, 15],
        "gamma": [0.01, 0.05]
    }

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))


    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y_%m_%d_%H_%M_%S')
    df_output_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/results_{}.csv".format(timestamp)

    def output_csv():
        df = pd.DataFrame(results)
        df.to_csv(df_output_path, index=False, encoding='utf-8')


    results = []
    for ix, p in enumerate(dict_product(pranges)):
        if ix > 185:
            print("Parameter combination: {}".format(ix))
            model_p, acc_p, time_p = run(**p)
            pout = copy.deepcopy(p)
            pout['acc'] = acc_p
            pout['time'] = time_p
            results.append(pout)
            if ix % 10 == 0:
                output_csv()



    output_csv()

    # run_type = 2

    # n_epochs = 2

    # n_input_channels = 8

    # # -----------------------------------------------------------------------------

    # if run_type in [1, 3]:

    #     print("\nfine tuning:\n")

    #     model_ft = resnet.resnet18(pretrained=True, n_input_channels=n_input_channels)

    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, ncats)

    #     model_ft = model_ft.to(device)

    #     criterion = nn.CrossEntropyLoss()

    #     # Observe that all parameters are being optimized
    #     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    #     # Decay LR by a factor of 0.1 every 7 epochs
    #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


    #     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                            num_epochs=n_epochs)

    # # -----------------------------------------------------------------------------

    # if run_type in [2, 3]:

    #     print("\nfixed feature extractor:\n")

    #     model_conv = resnet.resnet18(pretrained=True, n_input_channels=n_input_channels)
    #     for param in model_conv.parameters():
    #         param.requires_grad = False

    #     # Parameters of newly constructed modules have requires_grad=True by default
    #     num_ftrs = model_conv.fc.in_features

    #     model_conv.fc = nn.Linear(num_ftrs, ncats)


    #     model_conv = model_conv.to(device)

    #     criterion = nn.CrossEntropyLoss()

    #     # Observe that only parameters of final layer are being optimized as
    #     # opoosed to before.
    #     optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    #     # Decay LR by a factor of 0.1 every 7 epochs
    #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)


    #     model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                              exp_lr_scheduler, num_epochs=n_epochs)



