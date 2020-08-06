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
from data_prep import *


# *****************
# *****************
# json_path = "settings/nigeria_acled.json"
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

static_hash = s.build_hash(s.static)

device = torch.device("cuda:{}".format(s.config["cuda_device_id"]) if torch.cuda.is_available() else "cpu")
print("\nRunning on:", device)

# -----------------------------------------------------------------------------

sample_data = None

for ix, (param_hash, params) in enumerate(tasks):

    print('\n' + '-' * 40)
    print("\nParameter combination: {}/{}".format(ix+1, s.param_count))
    print("\nParam hash: {}\n".format(param_hash))

    state_path = os.path.join(base_path, "output/s1_state/state_{}_{}.pt".format(param_hash, s.config["version"]))

    # ====================
    # ====================
    # TRAIN
    # ====================
    # ====================
    if (s.config["run"]["train"]) and (not os.path.isfile(state_path) or s.config["overwrite_train"]):

        if sample_data is None:
            ps = PrepareSamples(s.base_path, s.static, static_hash, s.config["version"], overwrite=s.config["overwrite_sample_prep"])
            sample_data, class_sizes = ps.run()
            ps.print_counts()


        params["train"] = {}
        params["train"]['ncats'] = len(ps.cat_names)
        params["train"]["train_class_sizes"] = class_sizes["train"]
        params["train"]["val_class_sizes"] = class_sizes["val"]


        dataloaders = build_dataloaders(
            sample_data,
            base_path,
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

        # if s.config["run"]["train"]:
        train_cnn.init_loss()
        acc_p, class_p, time_p = train_cnn.train()
        params["train"]["acc"] = acc_p
        params["train"]["class_acc"] = class_p
        params["train"]["time"] = time_p
        s.write_to_json(param_hash, params)
        train_cnn.save(state_path)
        # else:
            # train_cnn.load(state_path)


        # if s.config["run"]["test"]:
        #     train_cnn.init_loss()
        #     epoch_loss, epoch_acc, class_acc, time_elapsed = train_cnn.test()

        # if s.config["run"]["predict"]:
        #     pred_data, _ = train_cnn.predict(features=True)
        #     # old line from custom_predict section to use predict from sample data manually,
        #     # which would involve then building a new dataloader. Just leaving this line and
        #     # comment here as reminder for potential future implementation
        #     # predict_df = sample_data["predict"].reset_index()


    # ====================
    # ====================
    # PREDICT
    # ====================
    # ====================

    if (s.config["run"]["custom_predict"]):

        for predict_key in s.data["predict"].keys():

            predict_settings = s.data["predict"][predict_key]
            predict_hash = s.build_hash(predict_settings, nchar=7)

            fbasename = "predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash, s.config["version"], s.config["predict_tag"])

            raw_out_path = os.path.join(base_path, "output/s1_predict", "raw_" + fbasename)
            group_out_path = os.path.join(base_path, "output/s1_predict", fbasename)

            if not os.path.isfile(group_out_path) or s.config["overwrite_predict"]:

                predict_df = prepare_sample(base_path, predict_key, predict_settings)

                # if os.path.isfile(predict_settings["sample"]):
                #     predict_df = pd.read_csv(predict_settings["sample"], quotechar='\"',
                #                             na_values='', keep_default_na=False,
                #                             encoding='utf-8')
                # else:
                #     try:
                #         predict_src = SurveyData(base_path)
                #         predict_df = predict_src.load(predict_settings["sample"])
                #     except:
                #         raise ValueError("Invalid predict sample id: `{}`".format(predict_settings["sample"]))

                predict_data = {"predict": predict_df}

                new_dataloaders = build_dataloaders(
                    predict_data,
                    base_path,
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
                new_pred_data, new_proba_data, new_feats_data, _ = new_cnn.predict()

                pred_dicts = [{"pred_class": i} for i in new_pred_data]

                proba_labels = ["proba_{}_{}".format(i, j) for i, j in enumerate(s.static["cat_names"])]
                proba_dicts = [dict(zip(proba_labels, i)) for i in new_proba_data]

                feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]
                feat_dicts = [dict(zip(feat_labels, i)) for i in new_feats_data]

                results_labels = ["pred_class"] + proba_labels + feat_labels

                # python 2.7
                results_dicts = []
                for i in xrange(len(pred_dicts)):
                    tmp = {}
                    tmp.update(pred_dicts[i])
                    tmp.update(proba_dicts[i])
                    tmp.update(feat_dicts[i])
                    results_dicts.append(tmp)

                # python 3.5+
                # results_dicts = [{**pred_dicts[i], **proba_dicts[i], **feat_dicts[i]} for i in xrange(len(pred_dicts))]

                pred_df = pd.DataFrame(results_dicts)

                full_out = predict_data["predict"].merge(pred_df, left_index=True, right_index=True)
                full_col_order = list(predict_data["predict"].columns) + results_labels
                full_out = full_out[full_col_order]
                full_out.to_csv(raw_out_path, index=False, encoding='utf-8')

                # aggregate by group
                if "group" in full_col_order:
                    agg_fields = {}
                    for i in full_col_order:
                        if i.startswith("feat"):
                            agg_fields[i] = "mean"
                        elif i.startswith("proba"):
                            agg_fields[i] = "mean"
                        elif i.startswith("pred"):
                            agg_fields[i] = pd.Series.mode
                        else:
                            agg_fields[i] = "last"

                    del agg_fields["group"]
                    group_out = full_out.groupby("group").agg(agg_fields).reset_index()
                    group_col_order = [i for i in full_col_order if i != "group"]
                    group_out = group_out[group_col_order]
                    group_out.to_csv(group_out_path, index=False, encoding='utf-8')
                else:
                    full_out.to_csv(group_out_path, index=False, encoding='utf-8')
