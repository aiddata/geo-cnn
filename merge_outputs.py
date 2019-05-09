
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
json_path = "settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

s = Settings()
s.load(json_path)
base_path = s.base_path


version = s.config["version"]
predict_tag = s.config["predict_tag"]
model_tag = s.config["model_tag"]

# final merged output
merge_out_path = os.path.join(base_path, "output/s2_merge/merge_{}_{}_{}_{}.csv".format(version, predict_tag, model_tag, timestamp))

# find input data based on models
#   use combinations of version, predict tag, and model tag to search
regex_str = os.path.join(base_path, "output/s2_metrics/metrics_*_{}_{}_{}.csv".format(version, predict_tag, model_tag))
regex_search = glob.glob(regex_str)

qlist = regex_search



merge_df_list = []

for model_file in qlist:
    df = pd.read_csv(model_file, quotechar='\"',
                     na_values='', keep_default_na=False,
                     encoding='utf-8')
    merge_df_list.append(df)
    model_hash = os.path.basename(model_file)[:-4].split("_")[1]
    param_json_path = os.path.join(base_path, "output/s1_train/train_{}_{}.json".format(model_hash, version))
    with open(param_json_path) as f:
        params = json.load(f)
        for k in params:
            if k in ["train", "static"]:
                for ki in params[k]:
                    df[ki] = [params[k][ki]] * len(df)
            else:
                df[k] = [params[k]] * len(df)


merge_df = pd.concat(merge_df_list, axis=0, ignore_index=True)


merge_df.to_csv(merge_out_path, index=False, encoding='utf-8')
