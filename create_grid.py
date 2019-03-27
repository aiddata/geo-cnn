
import sys
import os
import json
import time
import copy

import fiona
from shapely.geometry import shape, Point
from shapely.prepared import prep
from shapely.ops import cascaded_union

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


    def gen_grid(self, pixel_size, quiet=False):

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
                    "cell_id": cell_id,
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
