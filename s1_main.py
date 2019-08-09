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

output_dirs = ["s0_settings", "s1_params", "s1_state", "s1_predict", "s1_train", "s2_metrics", "s2_models", "s2_merge", "s3_s1_predict", "s3_s2_predict"]
for d in output_dirs:
    abs_d = os.path.join(base_path, "output", d)
    make_dir(abs_d)

job_dir = os.path.basename(os.path.dirname(json_path))
shutil.copyfile(
    json_path,
    os.path.join(base_path, "output/s0_settings/settings_{}_{}.json".format(job_dir, timestamp))
)

s.save_params()
tasks = s.hashed_iter()

predict_settings = s.data[s.config["predict"]]
predict_hash = s.build_hash(predict_settings, nchar=7)

device = torch.device("cuda:{}".format(s.config["cuda_device_id"]) if torch.cuda.is_available() else "cpu")
print("\nRunning on:", device)

# -----------------------------------------------------------------------------

sample_data = None
predict_data = None

for ix, (param_hash, params) in enumerate(tasks):

    print('\n' + '-' * 40)
    print("\nParameter combination: {}/{}".format(ix+1, s.param_count))
    print("\nParam hash: {}\n".format(param_hash))

    state_path = os.path.join(base_path, "output/s1_state/state_{}_{}.pt".format(param_hash, s.config["version"]))

    full_out_path = os.path.join(base_path, "output/s1_predict/raw_predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash, s.config["version"], s.config["predict_tag"]))
    group_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash, s.config["version"], s.config["predict_tag"]))

    custom_out_path = os.path.join(base_path, "output/s1_predict/predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash, s.config["version"], s.config["predict_tag"]))


    if (not os.path.isfile(state_path) or s.config["overwrite_train"]) and (s.config["run"]["train"] or s.config["run"]["test"] or s.config["run"]["predict"]):

        if sample_data is None:
            ps = PrepareSamples(s.base_path, s.static, s.config["version"], overwrite=s.config["overwrite_sample_prep"])
            sample_data, class_sizes = ps.run()
            ps.print_counts()

        params["train"] = {}
        params["train"]['ncats'] = len(ps.cat_names)
        params["train"]["train_class_sizes"] = class_sizes["train"]
        params["train"]["val_class_sizes"] = class_sizes["val"]

        dataloaders = build_dataloaders(
            sample_data,
            base_path,
            params["static"]["imagery_year"],
            params["static"]["imagery_type"],
            params["static"]["imagery_bands"],
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"])

        train_cnn = RunCNN(dataloaders, device, parallel=False, **params)

        train_cnn.init_training()
        train_cnn.init_print()
        train_cnn.init_net()

        if s.config["run"]["train"]:
            train_cnn.init_loss()
            acc_p, class_p, time_p = train_cnn.train()
            params["train"]["acc"] = acc_p
            params["train"]["class_acc"] = class_p
            params["train"]["time"] = time_p
            s.write_to_json(param_hash, params)
            train_cnn.save(state_path)
        else:
            train_cnn.load(state_path)

        # if s.config["run"]["test"]:
        #     train_cnn.init_loss()
        #     epoch_loss, epoch_acc, class_acc, time_elapsed = train_cnn.test()

        # if s.config["run"]["predict"]:
        #     pred_data, _ = train_cnn.predict(features=True)


    if (not os.path.isfile(full_out_path) or s.config["overwrite_survey_predict"]) and (s.config["run"]["survey_predict"]):

        # load survey data
        if predict_data is None:
            survey_data = SurveyData(base_path, predict_settings)
            predict_data = {
                "predict": survey_data.surveys[predict_settings["survey"]].copy(deep=True)
            }

        new_dataloaders = build_dataloaders(
            predict_data,
            base_path,
            predict_settings["imagery_year"],
            params["static"]["imagery_type"],
            params["static"]["imagery_bands"],
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
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
        full_out.to_csv(full_out_path, index=False, encoding='utf-8')

        # aggregate by group
        if "group" in full_col_order:
            agg_fields = {i:"mean" if i.startswith("feat") else "last" for i in full_col_order}
            del agg_fields["group"]
            group_out = full_out.groupby("group").agg(agg_fields).reset_index()
            group_col_order = [i for i in full_col_order if i != "group"]
            group_out = group_out[group_col_order]
            group_out.to_csv(group_out_path, index=False, encoding='utf-8')



    if (not os.path.isfile(custom_out_path) or s.config["overwrite_custom_predict"]) and (s.config["run"]["custom_predict"]):

        # load custom data
        if predict_data is None:
            custom_data = pd.read_csv(predict_settings["data"], quotechar='\"',
                                     na_values='', keep_default_na=False,
                                     encoding='utf-8')
            predict_data = {
                "predict": custom_data
            }

        new_dataloaders = build_dataloaders(
            predict_data,
            base_path,
            predict_settings["imagery_year"],
            predict_settings["imagery_year"],
            predict_settings["imagery_year"],
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)

        new_cnn = RunCNN(new_dataloaders, device, parallel=False, **params)
        new_cnn.load(state_path)

        # predict
        new_pred_data, _ = new_cnn.predict(features=True)

        # merge predict with original data
        feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]
        pred_dicts = [dict(zip(feat_labels, i)) for i in new_pred_data]
        pred_df = pd.DataFrame(pred_dicts)
        custom_out = predict_data["predict"].merge(pred_df, left_index=True, right_index=True)
        full_col_order = list(predict_data["predict"].columns) + feat_labels
        custom_out = custom_out[full_col_order]
        custom_out.to_csv(custom_out_path, index=False, encoding='utf-8')
