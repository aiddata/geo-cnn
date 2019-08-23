
import sys
import os
import json
import time
import copy
import warnings
import itertools

import fiona
from shapely.geometry import shape, Point
from shapely.prepared import prep
from shapely.ops import cascaded_union

import numpy as np
import pandas as pd
import geopandas as gpd



class PointGrid():
    """ Generate a point grid given a boundary and pixel size
    """

    def __init__(self, boundary_geom):
        bnds = boundary_geom.bounds
        (self.minx, self.miny, self.maxx, self.maxy) = bnds
        self.prep_shape = prep(cascaded_union([shape(f['geometry']) for f in boundary_geom]))
        self.prop_list = None
        self.pixel_size = None
        self.size = None
        self.df = None
        self.gdf = None


    def gen_grid(self, pixel_size, quiet=False):

        self.pixel_size = pixel_size
        xsize = pixel_size
        ysize = pixel_size

        self.maxx = self.maxx - xsize
        self.miny = self.miny + ysize

        ncols = int((self.maxx - self.minx) / xsize) + 1
        nrows = int((self.maxy - self.miny) / ysize) + 1

        # start in top left and go row by row
        for r in xrange(nrows):
            y = self.maxy - (r * ysize)

            for c in xrange(ncols):
                x = self.minx + (c * xsize)

                if not self.prep_shape.contains(Point(x, y)):
                    continue

                cell_id = (r * ncols) + c

                props = {
                    "sample_id": cell_id,
                    "row": r,
                    "column": c,
                    "lon": x,
                    "lat": y
                }

                yield props


    def grid(self, pixel_size, **kwargs):
        self.prop_list = list(self.gen_grid(pixel_size, **kwargs))


    def grid_size(self):
        if self.prop_list is None:
            raise Exception("Generate grid before running `size`")
        self.size = len(self.prop_list)
        return self.size


    def to_geojson(self, path):
        if self.prop_list is None:
            raise Exception("Generate grid before running `to_geojson`")

        feature_list = []

        for props in self.prop_list:
            geom = {
                "type": "Point",
                "coordinates": [props['lon'], props['lat']]
            }

            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": geom
            }

            feature_list.append(feature)

        geo_out = {
            "type": "FeatureCollection",
            "features": feature_list
        }

        geo_file = open(path, "w")
        json.dump(geo_out, geo_file)
        geo_file.close()


    def to_csv(self, path):
        if not hasattr(self, 'df'):
            self.df = self.to_dataframe()
        self.df.to_csv(path, index=False, encoding='utf-8')


    def to_dataframe(self):
        if self.prop_list is None:
            raise Exception("Generate grid before running `to_dataframe`")
        self.df = pd.DataFrame(self.prop_list)
        return self.df


    def to_geodataframe(self):
        if self.prop_list is None:
            raise Exception("Generate grid before running `to_geodataframe`")
        if hasattr(self, 'df'):
            df = copy.deepcopy(self.df)
        else:
            df = self.to_dataframe()
        df['geometry'] = df.apply(lambda z: Point(z['lon'], z['lat']), axis=1)
        self.gdf = gpd.GeoDataFrame(df)
        return self.gdf


class SampleFill():

    def __init__(self, df):
        self.df = df


    def rfill(self, nfill, distance, dict, mode):
        """nfill
        add new points unrelated to original points

        new points will inherit dict provide

        new points will be buffer distance away from original points
        """
        rfill_list = []
        gdf = self.df.copy(deep=True)
        gdf["geometry"] = gdf.apply(lambda x: Point(x.lon, x.lat), axis=1)
        gdf = gpd.GeoDataFrame(gdf)
        gdf = gdf.buffer(distance)
        bnds = gdf.total_bounds
        import random
        from shapely.ops import cascaded_union as dissolve
        from shapely.prepared import prep
        dissolved_geom = prep(dissolve(gdf.geometry))
        while len(rfill_list) <= nfill:
            rlon = random.uniform(bnds[0], bnds[2])
            rlat = random.uniform(bnds[1], bnds[3])
            rpoint = Point(rlon, rlat)
            if not dissolved_geom.contains(rpoint):
                new_sample = dict.copy()
                new_sample["group"] = "rfill"
                rfill_list.append(new_sample)
        tmp_df = pd.DataFrame(rfill_list)
        return tmp_df


    def gfill(self, nfill, distance, mode="fixed"):
        """group fill
        adds additional points grouped with original points

        new points will inherit row attributes of parent
        """
        self.df["group"] = "orig"

        if mode in [None, "None", "none", 0, "False", "false", False]:
            return
        elif nfill == 0 or distance == 0:
            warnings.warn("SampleFill: nfill or distance set to 0, sample will not be filled")
            return
        elif mode == "fixed":
            tmp_df = self._gfill_fixed(nfill, distance=distance)
        elif mode == "random":
            tmp_df = self._gfill_random(nfill, distance=distance)
        else:
            raise ValueError("SampleFill: Invalid fill mode ({})".format(mode))

        self.df = pd.concat([self.df, tmp_df], ignore_index=True, sort=False)


    def _gfill_fixed(self, nfill, distance):
        """
        nfill (int)
            number of additional points to add within given bounds for each
            grid point
        distance (float)
            maximum decimal degree distance from original grid point (lon, lat)
            allowed in each direction
        """
        fill_list = []

        for i, parent in self.df.iterrows():

            lon_vals = np.linspace(parent["lon"] - distance, parent["lon"] + distance, np.ceil(np.sqrt(nfill)))
            lat_vals = np.linspace(parent["lat"] - distance, parent["lat"] + distance, np.ceil(np.sqrt(nfill)))
            sub_grid = list(itertools.product(lon_vals, lat_vals))

            tmp_fill_list = []
            for j in sub_grid:
                child = parent.to_dict()
                child["group"] = "fill"
                child["lon"], child["lat"] = j
                tmp_fill_list.append(child)
            fill_list.extend(tmp_fill_list)

        tmp_df = pd.DataFrame(fill_list)
        return tmp_df


    def _gfill_random(self, nfill, distance):
        """
        nfill (int)
            number of additional points to add within given bounds for each
            grid point
        distance (float)
            maximum decimal degree distance from original grid point (lon, lat)
            allowed in each direction
        """
        fill_list = []
        for i, parent in self.df.iterrows():
            tmp_fill_list = []
            for j in range(nfill):
                child = parent.to_dict()
                child["group"] = "fill"
                child["lon"] = child["lon"] + np.random.uniform(-distance, distance)
                child["lat"] = child["lat"] + np.random.uniform(-distance, distance)
                tmp_fill_list.append(child)
            fill_list.extend(tmp_fill_list)
        tmp_df = pd.DataFrame(fill_list)
        return tmp_df
