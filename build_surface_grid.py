
from __future__ import print_function, division

import os
import copy
import shutil
import datetime
import time

import fiona
import pandas as pd

from settings_builder import Settings
from data_prep import make_dir

from create_grid import PointGrid
from load_data import NTL_Reader

# *****************
# *****************
json_path = "settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************

s = Settings()
s.load(json_path)

output_dirs = ["s3_grid"]
for d in output_dirs:
    abs_d = os.path.join(s.base_path, "output", d)
    make_dir(abs_d)


boundary_path = s.data["static"]["boundary_path"]

pixel_size = s.data["surface"]["pixel_size"]

ntl_calibrated = s.data["surface"]["ntl_calibrated"]
ntl_year = s.data["surface"]["ntl_year"]
ntl_dim = s.data["surface"]["ntl_dim"]

grid_path = os.path.join(s.base_path, "s3_grid/grid_{}.csv".format("???"))

# -----------------------------------------------------------------------------


boundary_src = fiona.open(boundary_path, "r")

grid = PointGrid(boundary_src)

boundary_src.close()

grid.grid(pixel_size)

grid.df = grid.to_dataframe()

ntl = NTL_Reader(calibrated=ntl_calibrated)
ntl.set_year(ntl_year)

grid.df['ntl'] = grid.df.apply(lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)

grid.to_csv(grid_path)
