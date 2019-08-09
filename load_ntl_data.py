
import os
import glob
import rasterio


class NTL_Reader():
    """ Read in NTL data
    """

    def __init__(self, ntl_type, calibrated=False):
        self.ntl_type = ntl_type
        if ntl_type == "dmsp":
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites"
            if calibrated:
                self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
            self.fregex = os.path.join(self.base, "F*{}.*.tif")
            self.default_dim = 7
        elif ntl_type == "viirs":
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/viirs/vcmcfg_dnb_composites_v10/yearly/max/"
            self.fregex = os.path.join(self.base, "{}_*.tif")
            self.default_dim = 14
        else:
            raise ValueError("NTL_Reader: invalid ntl_type `{}`".format(ntl_type))
        self.year = None
        self.path = None
        self.file = None

    def set_year(self, year):
        if self.file is not None:
            self.file.close()
        self.year = year
        try:
            self.path = glob.glob(self.fregex.format(self.year))[0]
        except:
            print(self.fregex.format(self.year))
            raise
        self.file = rasterio.open(self.path)

    def value(self, lon, lat, ntl_dim=-1, method="mean"):
        """Get nighttime lights average value for grid around point
        """
        if ntl_dim == -1:
            ntl_dim = self.default_dim
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
