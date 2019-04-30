"""

qsub -I -l nodes=1:hima:gpu:ppn=64 -l walltime=72:00:00

hi06 = p100 x2
hi07 = v100 x2



"""

from __future__ import print_function, division

import os
import copy
# import datetime
# import time

import pandas as pd
import torch

from load_data import build_dataloaders
from runscript import RunCNN
from load_survey_data import SurveyData
from settings_builder import Settings
from data_prep import make_dir, gen_sample_size, apply_types, normalize, PrepareSamples


# *****************
# *****************
json_path = "settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


print('-' * 40)
print("\nInitializing...")

# timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
#     '%Y_%m_%d_%H_%M_%S')

# date_str = datetime.datetime.now().strftime("%Y%m%d")

s = Settings()
s.load(json_path)
base_path = s.base_path
s.set_param_count()

output_dirs = ["s1_params", "s1_state", "s1_predict", "s1_train", "s2_models", "s2_merge"]
for d in output_dirs:
    abs_d = os.path.join(base_path, "output", d)
    make_dir(abs_d)

s.save_params()
tasks = s.hashed_iter()

survey_data = SurveyData(base_path, s.static)

if s.config["run"]["train"] or s.config["run"]["test"] or s.config["run"]["predict"]:
    ps = PrepareSamples(s.base_path, s.static, s.config["version"], overwrite=s.config["overwrite_sample_prep"])
    dataframe_dict, class_sizes = ps.run()
    ps.print_counts()

device = torch.device("cuda:{}".format(s.config["cuda_device_id"]) if torch.cuda.is_available() else "cpu")
print("\nRunning on:", device)

# -----------------------------------------------------------------------------

for ix, (param_hash, params) in enumerate(tasks):
    print('\n' + '-' * 40)
    print("\nParameter combination: {}/{}".format(ix+1, s.param_count))
    print("\nParam hash: {}\n".format(param_hash))
    state_path = os.path.join(base_path, "output/s1_state/state_{}_{}.pt".format(param_hash, s.config["version"]))
    new_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}_{}.csv".format(param_hash, s.config["version"], s.config["predict_tag"]))
    # -----------------
    if (not os.path.isfile(state_path) or s.config["overwrite_train"]) and (s.config["run"]["train"] or s.config["run"]["test"] or s.config["run"]["predict"]):
        params["train"] = {}
        params["train"]['ncats'] = len(ps.cat_names)
        params["train"]["train_class_sizes"] = class_sizes["train"]
        params["train"]["val_class_sizes"] = class_sizes["val"]
        dataloaders = build_dataloaders(
            dataframe_dict,
            base_path,
            params["static"]["imagery_year"],
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"])
        train_cnn = RunCNN(
            dataloaders, device, parallel=False, **params)
        if s.config["run"]["train"]:
            acc_p, class_p, time_p = train_cnn.train()
            params["train"]["acc"] = acc_p
            params["train"]["class_acc"] = class_p
            params["train"]["time"] = time_p
            s.write_to_json(param_hash, params)
            train_cnn.save(state_path)
        else:
            train_cnn.load(state_path)
        if s.config["run"]["test"]:
            epoch_loss, epoch_acc, class_acc, time_elapsed = train_cnn.test()
        if s.config["run"]["predict"]:
            pred_data, _ = train_cnn.predict(features=True)
    # -----------------
    if (not os.path.isfile(new_out_path) or s.config["overwrite_predict_new"]) and (s.config["run"]["predict_new"]):
        """
        - load data
        - load trained cnn state
        - run predict
        - append cnn features to original data
        - output to csv for second stage models
        """
        new_data = {
            "predict": survey_data.surveys[params["static"]["survey"]].copy(deep=True)
        }
        new_dataloaders = build_dataloaders(
            new_data,
            base_path,
            params["static"]["imagery_year"],
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)
        new_cnn = RunCNN(
            new_dataloaders, device, parallel=False, **params)
        new_cnn.load(state_path)
        # ---------
        new_pred_data, _ = new_cnn.predict(features=True)
        feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]
        pred_dicts = [dict(zip(feat_labels, i)) for i in new_pred_data]
        pred_df = pd.DataFrame(pred_dicts)
        new_out = new_data["predict"].merge(pred_df, left_index=True, right_index=True)
        col_order = list(new_data["predict"].columns) + feat_labels
        new_out = new_out[col_order]
        new_out.to_csv(new_out_path, index=False, encoding='utf-8')
