

import os

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
model_list = s3_info["predict"]["models"]
input_list = s3_info["predict"]["inputs"]

model_tag = s.config["model_tag"]
surface_tag = s.config["surface_tag"]


tasks = s.hashed_iter()

qlist = []

for ix, (param_hash, params) in enumerate(tasks):
    for model_name in model_list:
        s3_s2_string = "_".join(str(i) for i in [
            model_name,
            param_hash,
            predict_hash,
            s3_info["grid"]["boundary_id"],
            s3_info["predict"]["imagery_year"],
            s.config["version"],
            s.config["predict_tag"],
            s.config["model_tag"]
        ])
        s3_s2_path = os.path.join(s.base_path, "output/s3_s2_predict/predict_{}.csv".format(s3_s2_string))
        for input_name in input_list:
            s4_path = os.path.join(s.base_path, "output/s4_surface/surface_{}_{}_{}.tif".format(input_name, s3_s2_string, surface_tag))
            df = pd.read_csv(s3_s2_path)
            shape = (max(df.row), max(df.column))
            blank = np.zeros(shape)
            for i, row in df.iterrows():
                blank[row.row-1, row.column-1] = row[input_name]
            xmin, ymax = min(df.lon), max(df.lat)
            pixel_size =  s3_info["surface"]["pixel_size"]
            meta = {}
            meta["transform"] = Affine(pixel_size, 0, xmin,
                                       0, -pixel_size, ymax)
            meta['height'] = shape[0]
            meta['width'] = shape[1]
            meta['driver'] = 'GTiff'
            meta['count'] = 1
            meta['dtype'] = str(blank.dtype)
            with rasterio.open(s4_path, 'w', **meta) as result:
                result.write(np.array([blank]))
