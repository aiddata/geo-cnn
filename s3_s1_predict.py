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
json_path = "settings/tanzania_2010_dhs.json"
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

output_dirs = ["s0_settings", "s1_params", "s1_state", "s1_predict", "s1_train", "s2_metrics", "s2_models", "s2_merge"]
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

predict_settings = s.data["custom_predict"]
predict_hash = s.build_hash(predict_settings, nchar=7)

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

    custom_out_path = os.path.join(base_path, "output/s3_s1_predict/predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash, s.config["version"], s.config["predict_tag"]))

    if (not os.path.isfile(custom_out_path) or s.config["overwrite_custom_predict"]):

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
            data_transform=None,
            dim=params["dim"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            agg_method=params["agg_method"],
            shuffle=False)


        new_cnn = RunCNN(new_dataloaders, device, parallel=False, **params)

        new_cnn.init_training()
        new_cnn.init_net()
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
