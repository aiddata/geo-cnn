
import os
import rasterio
import numpy as np
from torch.utils.data import Dataset


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

        label = None
        if 'label' in row
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
