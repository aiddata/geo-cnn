
import os
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def build_dataloaders(df_dict, base_path, data_transform=None, dim=224, batch_size=64, num_workers=16, agg_method="max", shuffle=True):

    if data_transform == None:

        data_transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # imagenet means
                                 std=[0.229, 0.224, 0.225]), # imagenet stds
        ])


    dataloaders = {}

    # where group is train, val, test, predict
    for group in df_dict:
        tmp_dset = BandDataset(df_dict[group], base_path, dim=dim, transform=data_transform, agg_method=agg_method)
        dataloaders[group] = DataLoader(tmp_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    # train_dset = BandDataset(df_dict["train"], base_path, dim=dim, transform=data_transform, agg_method=agg_method)
    # val_dset = BandDataset(df_dict["val"], base_path, dim=dim, transform=data_transform, agg_method=agg_method)
    # test_dset = BandDataset(df_dict["test"], base_path, dim=dim, transform=data_transform, agg_method=agg_method)
    # predict_dset = BandDataset(df_dict["predict"], base_path, dim=dim, transform=data_transform, agg_method=agg_method)

    # train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # predict_dataloader = DataLoader(predict_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # dataloaders = {
    #     "train": train_dataloader,
    #     "val": val_dataloader,
    #     "test": test_dataloader,
    #     "predict": predict_dataloader
    # }

    return dataloaders



class BandDataset(Dataset):
    """Get the data
    """
    def __init__(self, dataframe, root_dir, dim=224, transform=None, agg_method="max"):

        self.dim = dim
        self.year = 2010
        # self.bands = ["b1", "b2", "b3", ]
        self.bands = ["b1", "b2", "b3", "b4", "b5", "b7", "b61", "b62"]
        # self.bands = ["b3", "b4", "b5"]

        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

        self.agg_method = agg_method


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        dim = self.dim

        row = self.dataframe.iloc[idx]

        # label = None
        label = -1
        if 'label' in row:
            label = row['label']

        lon = row['lon']
        lat = row['lat']

        feature = np.empty((len(self.bands), dim, dim))

        for bnum, band in enumerate(self.bands):

            season_mosaics_path = os.path.join(
                self.root_dir, "landsat/data/mosaics/{0}_all", self.agg_method,
                "{0}_all_{1}.tif".format(self.year, band))

            season_mosaics = rasterio.open(season_mosaics_path)

            r, c = season_mosaics.index(lon, lat)
            win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
            data = season_mosaics.read(1, window=win)

            if data.shape != (dim, dim):
                raise Exception("bad feature (dim: ({0}, {0}), data shape: {1}".format(dim, data.shape))

            feature[bnum] = data

        if self.transform:
            feature = np.transpose(feature,(1,2,0))
            feature = self.transform(feature)

        # return torch.from_numpy(feature).float(), label
        return feature.float(), label
