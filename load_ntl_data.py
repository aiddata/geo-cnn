
import os
import glob
import rasterio


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
