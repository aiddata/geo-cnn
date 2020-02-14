

import os
import itertools

import pandas as pd
import numpy as np
from affine import Affine
import rasterio

from settings_builder import Settings


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************
# json_path = "settings/ghana_2008_dhs.json"


s = Settings()
s.load(json_path)


predict_settings = s.data[s.config["predict"]]
predict_hash = s.build_hash(predict_settings, nchar=7)


s3_info = s.data["third_stage"]

model_tag = s.config["model_tag"]
surface_tag = s.config["surface_tag"]

tasks = s.hashed_iter()

qlist = []


# ----------
# load raster for nodata checks

root_dir = s.config["base_path"]
year = s.static["imagery_year"]

# these will be the same across all so choice does not matter
agg_method = "mean"
band = "b1"

season_mosaics_path = os.path.join(
    root_dir, "landsat/data/{}/mosaics/{}_all".format(s.static["imagery_type"], year), agg_method,
    "{0}_all_{1}.tif".format(year, band))

season_mosaics = rasterio.open(season_mosaics_path)
# ----------

print "-----"

input_stage = s3_info["surface"]["input_stage"]

if input_stage == "s2":
    model_list = s3_info["predict"]["class_models"] + s3_info["predict"]["proba_models"]
    input_list = s3_info["predict"]["inputs"]
    surface_list = itertools.product(model_list, input_list)

if input_stage == "s1":
    surface_list = [
        (0, s3_info["surface"]["value_type"])
    ]

else:
    raise ValueError("Surface input stage must be either `s1` or `s2`. (`{}` given)".format(input_stage))


for ix, (param_hash, params) in enumerate(tasks):

    for surface_item in surface_list:

        print "Running: {} - {}".format(param_hash, surface_item)

        if input_stage == "s2":

            model_name, input_name = surface_item

            input_string = "_".join(str(i) for i in [
                model_name,
                param_hash,
                predict_hash,
                s3_info["grid"]["boundary_id"],
                s3_info["predict"]["imagery_year"],
                s.config["version"],
                s.config["predict_tag"],
                s.config["model_tag"]
            ])

            input_path = os.path.join(s.base_path, "output/s3_s2_predict/predict_{}.csv".format(input_string))


        elif input_stage == "s1":

            _, input_name = surface_item

            input_string = "_".join(str(i) for i in [
                param_hash,
                predict_hash,
                s.config["version"],
                s.config["predict_tag"]
            ])

            input_path = os.path.join(s.base_path, "output/s1_predict/predict_{}.csv".format(input_string))


        surface_string = input_string + "_" + surface_tag
        s3_surface_path = os.path.join(s.base_path, "output/s3_surface/surface_{}_{}.tif".format(input_name, surface_string))

        df = pd.read_csv(input_path)
        shape = (max(df.row), max(df.column))
        blank = np.full(shape, s3_info["surface"]["nodata_val"])

        for i, row in df.iterrows():
            dim = s3_info["surface"]["dim"]
            r, c = season_mosaics.index(row.lon, row.lat)
            win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
            data = season_mosaics.read(1, window=win)

            if data.shape != (dim, dim):
                raise Exception("bad feature (dim: ({0}, {0}), data shape: {1}".format(dim, data.shape))

            val = row[input_name]

            # should this be checking if data == 0 or data == s3_info["surface"]["nodata_val"]?
            # and should set value if true be 0 or nodataval?
            if np.sum(data == 0) / float(data.size) > s3_info["surface"]["scene_max_nodata"]:
                val = 0

            blank[row.row-1, row.column-1] = val


        pixel_size =  s3_info["grid"]["pixel_size"]
        xmin, ymax = min(df.lon)-(0.5*pixel_size), max(df.lat)+(0.5*pixel_size)
        meta = {}
        meta["crs"] = rasterio.crs.CRS.from_epsg(4236)
        meta["transform"] = Affine(pixel_size, 0, xmin,
                                    0, -pixel_size, ymax)
        meta['height'] = shape[0]
        meta['width'] = shape[1]
        meta['driver'] = 'GTiff'
        meta['count'] = 1
        meta['nodata'] = s3_info["surface"]["nodata_val"]
        meta['dtype'] = str(blank.dtype)
        with rasterio.open(s3_surface_path, 'w', **meta) as result:
            result.write(np.array([blank]))
