
import os
import glob
import rasterio
import numpy as np
from torch.utils.data import Dataset


class NTL_Reader():
    """ Read in NTL data
    """

    def __init__(self, calibrated=False):
        self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites"
        if calibrated:
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
        self.year = None
        self.path = None
        self.file = None

    def set_year(self, year):
        if self.file is not None:
            self.file.close()
        self.year = year
        self.path = glob.glob(os.path.join(self.base, "*{0}*.tif".format(self.year)))[0]
        self.file = rasterio.open(self.path)

    def value(self, lon, lat, ntl_dim=7, method="mean"):
        """Get nighttime lights average value for grid around point
        """
        r, c = self.file.index(lon, lat)
        ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
        ntl_data = self.file.read(1, window=ntl_win)
        if method == "mean":
            ntl_val = ntl_data.mean()
        elif method == "max":
            ntl_val = ntl_data.max()
        elif method == "min":
            ntl_val = ntl_data.min()
        else:
            raise ValueError("NTL_Reader: invalid method given for calculating NTL value ({})".format(method))
        return ntl_val


class BandDataset(Dataset):
    """Get the data
    """
    def __init__(self, dataframe, root_dir, year, dim=224, transform=None, agg_method="mean"):

        self.dim = dim
        self.year = year
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
                self.root_dir, "landsat/data/mosaics/{0}_all".format(self.year), self.agg_method,
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
