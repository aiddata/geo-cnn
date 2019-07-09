

import os
# import glob
# import time
# import datetime
# import pandas as pd

from settings_builder import Settings

from model_predict import run_models, run_tasks


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************
# json_path = "settings/ghana_2008_dhs.json"


s = Settings()
s.load(json_path)
base_path = s.base_path

mode = s.config["second_stage_mode"]

predict_hash = s.build_hash(s.data[s.config["predict"]], nchar=7)

# timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
#     '%Y_%m_%d_%H_%M_%S')


s3_info = s.data["third_stage"]

model_list = s3_info["predict"]["models"]
model_inputs = s3_info["predict"]["inputs"]

model_tag = s.config["model_tag"]

predict_settings = s.data[s.config["predict"]]
predict_hash = s.build_hash(predict_settings, nchar=7)

tasks = s.hashed_iter()

qlist = []

for ix, (param_hash, params) in enumerate(tasks):
    grid_predict_id = "{}_{}".format(s3_info["grid"]["boundary_id"], s3_info["predict"]["imagery_year"])

    grid_id_string = "{}_{}_{}_{}".format(
        param_hash, grid_predict_id, s.config["version"], s.config["predict_tag"]
    )

    grid_predict_path = os.path.join(base_path, "output/s3_s1_predict/predict_{}.csv".format(grid_id_string))

    train_predict_id = predict_hash

    train_id_string = "{}_{}_{}_{}".format(
        param_hash, train_predict_id, s.config["version"], s.config["predict_tag"]
    )

    for name in model_list:
        joblib_path = os.path.join(base_path, "output/s2_models/models_{}_INPUT_{}_{}.joblib".format(
            name, train_id_string, model_tag))
        qlist.append((grid_predict_path, joblib_path))



run_tasks(tasks=qlist, func=run_models, args=s, mode=mode)
