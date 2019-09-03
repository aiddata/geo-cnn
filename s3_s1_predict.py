"""

qsub -I -l nodes=1:hima:gpu:ppn=64 -l walltime=72:00:00

hi06 = p100 x2
hi07 = v100 x2



"""

from __future__ import print_function, division

import os
import copy
import shutil
import datetime
import time

import pandas as pd
import torch

from runscript import RunCNN, build_dataloaders
from load_survey_data import SurveyData
from settings_builder import Settings
from data_prep import make_dir, gen_sample_size, apply_types, normalize, PrepareSamples


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


print('-' * 40)
print("\nInitializing...")

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

# date_str = datetime.datetime.now().strftime("%Y%m%d")

s = Settings()
s.load(json_path)
base_path = s.base_path
s.set_param_count()
s.build_dirs()


job_dir = os.path.basename(os.path.dirname(json_path))
shutil.copyfile(
    json_path,
    os.path.join(base_path, "output/s0_settings/settings_{}_{}.json".format(job_dir, timestamp))
)

s.save_params()
tasks = s.hashed_iter()

# predict_settings = s.data["custom_predict"]
# predict_hash = s.build_hash(predict_settings, nchar=7)

s3_info = s.data["third_stage"]
# predict_settings = s3_info["predict"]

# surface_tag = s.config["surface_tag"]
# boundary_path = s3_info["grid"]["boundary_path"]
# fname = ".".join(os.path.basename(boundary_path).split(".")[:-1])
# grid_path = os.path.join(s.base_path, "output/s3_grid/grid_{}_{}.csv".format(surface_tag, fname))


device = torch.device("cuda:{}".format(s.config["cuda_device_id"]) if torch.cuda.is_available() else "cpu")
print("\nRunning on:", device)

# -----------------------------------------------------------------------------

predict_data = None

for ix, (param_hash, params) in enumerate(tasks):
    # ix, (param_hash, params) = list(enumerate(tasks))[0]

    print('\n' + '-' * 40)
    print("\nParameter combination: {}/{}".format(ix+1, s.param_count))
    print("\nParam hash: {}\n".format(param_hash))

    state_path = os.path.join(base_path, "output/s1_state/state_{}_{}.pt".format(param_hash, s.config["version"]))

    fbasename = "predict_{}_{}_{}_{}_{}.csv".format(
        param_hash, s3_info["grid"]["boundary_id"], s3_info["predict"]["imagery_year"],
        s.config["version"], s.config["predict_tag"])

    raw_out_path = os.path.join(base_path, "output/s3_s1_predict", "raw_" + fbasename)
    group_out_path = os.path.join(base_path, "output/s3_s1_predict", fbasename)

    print(group_out_path)


    if (not os.path.isfile(group_out_path) or s.config["overwrite_predict"]):

        if predict_data is None and s.config["predict"] == "source_predict":
            predict_src = pd.read_csv(s3_info["predict"]["source_predict"], quotechar='\"',
                                        na_values='', keep_default_na=False,
                                        encoding='utf-8')
            predict_data = {
                "predict": predict_src
            }

        elif predict_data is None and s.config["predict"] == "survey_predict":

            predict_src = SurveyData(base_path, s3_info["predict"], s3_info["predict"]["survey_year"])
            predict_data = {
                "predict": predict_src.surveys[s3_info["predict"]["survey_predict"]].copy(deep=True)
            }

        elif predict_data is None:
            raise ValueError("Invalid predict class: `{}`".format(s3_info["predict"]["method"]))



        new_dataloaders = build_dataloaders(
            predict_data,
            base_path,
            s3_info["predict"]["imagery_year"],
            params["static"]["imagery_type"],
            params["static"]["imagery_bands"],
            data_transform=None,
            dim=params["dim"],
            batch_size=s3_info["predict"]["batch_size"],
            num_workers=s3_info["predict"]["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)


        new_cnn = RunCNN(new_dataloaders, device, parallel=False, **params)

        new_cnn.init_training()
        new_cnn.init_print()
        new_cnn.init_net()

        new_cnn.load(state_path)

        # predict
        new_pred_data, _ = new_cnn.predict(features=True)

        # merge predict with original data
        feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]
        pred_dicts = [dict(zip(feat_labels, i)) for i in new_pred_data]
        pred_df = pd.DataFrame(pred_dicts)
        full_out = predict_data["predict"].merge(pred_df, left_index=True, right_index=True)
        full_col_order = list(predict_data["predict"].columns) + feat_labels
        full_out = full_out[full_col_order]
        full_out.to_csv(raw_out_path, index=False, encoding='utf-8')

        # aggregate by group
        if "group" in full_col_order:
            agg_fields = {i:"mean" if i.startswith("feat") else "last" for i in full_col_order}
            del agg_fields["group"]
            group_out = full_out.groupby("group").agg(agg_fields).reset_index()
            group_col_order = [i for i in full_col_order if i != "group"]
            group_out = group_out[group_col_order]
            group_out.to_csv(group_out_path, index=False, encoding='utf-8')
        else:
            full_out.to_csv(group_out_path, index=False, encoding='utf-8')
