"""
Merge outputs of metrics from second stage model tests into a single file
for comparison.
"""

import os
import datetime
import time
import json
import glob
from copy import deepcopy

import numpy as np
import pandas as pd

from settings_builder import Settings

# -----------------------------------------------------------------------------


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

s = Settings()
s.load(json_path)
base_path = s.base_path


version = s.config["version"]

# final merged output
merge_out_path = os.path.join(base_path, "output/s1_merge/merge_{}_{}.csv".format(version, timestamp))

# find input data based on models version tag
regex_str = os.path.join(base_path, "output/s1_train/train_*_{}.json".format(version))
regex_search = glob.glob(regex_str)

qlist = regex_search


dict_list = []




for param_json_path in qlist:
    json_dict = {}
    model_hash = str(os.path.basename(param_json_path).split("_")[1])
    json_dict["hash"] = model_hash
    with open(param_json_path) as f:
        params = json.load(f)
        for k in params:
            if k in ["train", "static"]:
                for ki in params[k]:
                    json_dict[ki] = params[k][ki]
            else:
                json_dict[k] = params[k]
    dict_list.append(json_dict)


df = pd.DataFrame(dict_list)

cols = ["hash", "acc"]
for i in df.columns:
    if i not in cols:
        cols.append(i)

df = df[cols]


df.to_csv(merge_out_path, index=False, encoding='utf-8')
