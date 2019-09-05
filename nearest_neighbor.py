

import numpy as np
from scipy.spatial import cKDTree


def snap(val, interval):
    if interval > 1:
        raise ValueError("Interval must be less than one")
    return round(np.floor( val* 1/interval)) / (1/interval)


class NN():
    """Use KDTree to find NearestNeighbor values for a given location

    Given DataFrame must have "longitude" and "latitude" columns
    Values returned are based on field/column provided

    """

    def __init__(self, df, k=1, agg_func=None):

        self.df = df
        self.default_k = k
        self.default_agg_func = agg_func


    def snap_to(self, interval):

        self.df["original_longitude"] = list(self.df.longitude)
        self.df["original_latitude"] = list(self.df.latitude)

        self.df.longitude = self.df.longitude.apply(lambda x: snap(x, interval))
        self.df.latitude = self.df.latitude.apply(lambda x: snap(x, interval))


    def build_tree(self):
        self.input_geom = zip(self.df.longitude, self.df.latitude)

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

        min_dist = min_dist[0]
        min_index = min_index[0]

        # lon_vals, lat_vals = zip(*self.tree.data[min_index])
        lon_vals, lat_vals = zip(*[self.input_geom[i] for i in min_index])

        nn_rows = self.df.loc[(self.df.longitude.isin(lon_vals)) & (self.df.latitude.isin(lat_vals))]

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
