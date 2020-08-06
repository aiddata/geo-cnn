
from __future__ import print_function, division

import os
import copy
import shutil
import datetime
import time

import fiona
import pandas as pd

from utils.settings_builder import Settings
from utils.data_prep import make_dir

from utils.create_grid import PointGrid
# from utils.load_ntl_data import NTL_Reader

# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************

s = Settings()
s.load(json_path)
s.build_dirs()


boundary_path = os.path.join(s.base_path, "data/boundary", s.data["static"]["grid_boundary_file"])

pixel_size = s.data["third_stage"]["grid"]["pixel_size"]

# ntl_calibrated = s.data["third_stage"]["predict"]["ntl_calibrated"]
# ntl_year = s.data["third_stage"]["predict"]["ntl_year"]
# ntl_dim = s.data["third_stage"]["predict"]["ntl_dim"]

surface_tag = s.config["surface_tag"]
# fname = os.path.basename(json_path, ".json")
fname = ".".join(os.path.basename(boundary_path).split(".")[:-1])
grid_path = os.path.join(s.base_path, "output/s0_grid/grid_{}_{}_{}.csv".format(str(pixel_size).split(".")[1], surface_tag, fname))

# -----------------------------------------------------------------------------


boundary_src = fiona.open(boundary_path, "r")

grid = PointGrid(boundary_src)

boundary_src.close()

grid.grid(pixel_size)

grid.df = grid.to_dataframe()

# ntl = NTL_Reader(calibrated=ntl_calibrated)
# ntl.set_year(ntl_year)

# grid.df['ntl'] = grid.df.apply(lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)

grid.to_csv(grid_path)

print("Grid completed ({})".format(grid_path))
