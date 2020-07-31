
import os
import glob
import rasterio


class NTL_Reader():
    """ Read in NTL data
    """

    def __init__(self, ntl_type=None, calibrated=False, dim=None, year=None, min_val=None):
        self.ntl_type = ntl_type
        if ntl_type == "dmsp":
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites"
            if calibrated:
                self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
            self.fregex = os.path.join(self.base, "F*{}.*.tif")
            default_dim = 7
        elif ntl_type == "viirs":
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/viirs/vcmcfg_dnb_composites_v10/yearly/max/"
            self.fregex = os.path.join(self.base, "{}_*.tif")
            default_dim = 14
        else:
            raise ValueError("NTL_Reader: invalid ntl_type `{}`".format(ntl_type))

        self.dim = dim if dim is not None else default_dim

        self.year = None
        self.path = None
        self.file = None

        if year is not None:
            self.set_year(year)


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
            ntl_dim = self.dim
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


    def assign_df_values(self, df, ntl_dim=-1, method="mean", field="ntl"):
        """Assign NTL values to a DF

        DF must contain lon and lat fields
        Field name may be specified but defaults to "ntl"
        Dimensions of NTL data to use may be specified as can aggregation method
        """
        df[field] = df.apply(lambda z: self.value(z['lon'], z['lat'], ntl_dim=ntl_dim, method=method), axis=1)
        df = df.loc[df[field] >= self.ntl_min]
        # df.to_csv(example_path, index=False, encoding='utf-8')
        return df
