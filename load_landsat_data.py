
import os
import rasterio
import numpy as np
from torch.utils.data import Dataset


class BandDataset(Dataset):
    """Get the data
    """
    def __init__(self, dataframe, root_dir, imagery_type, imagery_bands, agg_method="mean", dim=224, transform=None):

        self.dataframe = dataframe
        self.root_dir = root_dir

        self.imagery_type = imagery_type
        # self.bands = ["b1", "b2", "b3", "b4", "b5", "b7", "b61", "b62"] # landsat7
        # self.bands = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b9", "b10", "b11"] # landsat8
        self.bands = imagery_bands
        self.agg_method = agg_method

        self.dim = dim
        self.transform = transform


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        dim = self.dim

        row = self.dataframe.iloc[idx]

        temporal = row["temporal"]

        # label = None
        label = -1
        if 'label' in row:
            label = row['label']

        lon = row['lon']
        lat = row['lat']

        feature = np.empty((len(self.bands), dim, dim))

        for bnum, band in enumerate(self.bands):

            season_mosaics_path = os.path.join(
                self.root_dir, "landsat/data/{}/mosaics/{}".format(self.imagery_type, temporal), self.agg_method,
                "{}_{}.tif".format(temporal, band))
            try:
                season_mosaics = rasterio.open(season_mosaics_path)
            except:
                print(season_mosaics_path)
                raise

            r, c = season_mosaics.index(lon, lat)
            win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
            try:
                data = season_mosaics.read(1, window=win)
            except:
                print(win)
                raise

            if data.shape != (dim, dim):
                raise Exception("bad feature (dim: ({0}, {0}), data shape: {1}".format(dim, data.shape))

            feature[bnum] = data

        if self.transform:
            feature = np.transpose(feature,(1,2,0))
            feature = self.transform(feature)

        # return torch.from_numpy(feature).float(), label
        return feature.float(), label
