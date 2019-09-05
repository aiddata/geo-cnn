

import numpy as np
from scipy.spatial import cKDTree


def snap(val, interval):
    if interval > 1:
        raise ValueError("Interval must be less than one")
    return round(np.floor( val* 1/interval)) / (1/interval)


class NN():
    """Use KDTree to find NearestNeighbor values for a given location

    Given DataFrame must have "lon" and "lat" columns
    Values returned are based on field/column provided

    """

    def __init__(self, df, k=1, agg_func=None):

        self.df = df
        self.default_k = k
        self.default_agg_func = agg_func

        if "lon" not in df.columns and "longitude" in df.columns:
            df["lon"] = df.longitude
        if "lat" not in df.columns and "latitude" in df.columns:
            df["lat"] = df.latitude


    def snap_to(self, interval):

        self.df["original_lon"] = list(self.df.lon)
        self.df["original_lat"] = list(self.df.lat)

        self.df.lon = self.df.lon.apply(lambda x: snap(x, interval))
        self.df.lat = self.df.lat.apply(lambda x: snap(x, interval))


    def build_tree(self):
        self.input_geom = zip(self.df.lon, self.df.lat)

        self.tree = cKDTree(
            data=self.input_geom,
            leafsize=64
        )

    def query(self, loc, field, k=None, agg_func=None):

        if k is None:
            k = self.default_k

        if agg_func is None:
            agg_func = self.default_agg_func

        if k > 1 and agg_func is None:
            raise Exception("An aggregation must be provided when k > 1")

        min_dist, min_index = self.tree.query([loc], k)

        # min_dist = min_dist[0]
        # min_index = min_index[0]

        # lon_vals, lat_vals = zip(*self.tree.data[min_index])
        lon_vals, lat_vals = zip(*[self.input_geom[i] for i in min_index])

        nn_rows = self.df.loc[(self.df.lon.isin(lon_vals)) & (self.df.lat.isin(lat_vals))]

        if len(nn_rows) == 0:
            raise Exception("No NN match could be found")
        elif agg_func is None:
            # if len(nn_rows) > 1:
            #     warnings.warn("More than one NN match found; using first match.")
            nn = nn_rows.iloc[0]
            val = nn[field]
        else:
            vals = nn_rows[field].tolist()
            val = agg_func(vals)

        return val
