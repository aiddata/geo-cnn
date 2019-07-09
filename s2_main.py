

import os
import glob
import time
import datetime
import pandas as pd

from settings_builder import Settings

from model_prep import (pearson_r2, ModelHelper,
                        run_cv, find_best_alpha, predict_inner_test_fold,
                        scale_features, train, train_and_predict,
                        run_models, run_tasks)


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


s = Settings()
s.load(json_path)
base_path = s.base_path

mode = s.config["second_stage_mode"]

predict_hash = s.build_hash(s.data[s.config["predict"]], nchar=7)

# timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
#     '%Y_%m_%d_%H_%M_%S')

mh = ModelHelper(settings=s)

tasks = s.hashed_iter()

qlist = []
for param_hash, _ in tasks:
    fname = "predict_{}_{}_{}_{}.csv".format(
        param_hash, predict_hash, s.config["version"], s.config["predict_tag"])

    qlist.append(
        "_".join(fname.split("_")[1:])[:-4] 
    )


# qlist = ["7a118a3_2019_03_28_12_48_37"]
# qlist = pd.read_csv(os.path.join(base_path, "cnn_results_merge_1.csv"))["id_string"].tolist()
# print qlist

run_tasks(tasks=qlist, func=run_models, args=mh, mode=mode)
