"""
specify survey layer (with year) to validate

load s4_surface for same year (use json or fixed args to determine which surface?)

compare point (or buffer zs?) value of surface to survey value for each survey point

output dataframe of survey value, point value, buffer value along with percent differences? (include lat/lon to map errors)

"""


import os
import itertools
import errno
import rasterio
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.settings_builder import Settings


import glob
import time
import datetime

timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def surface_win_mean(src, pnt, dim):
    r, c = src.index(pnt.lon, pnt.lat)
    win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
    data = src.read(1, window=win, boundless=True, masked=True)
    win_val = np.mean(data)
    return win_val


json_path_list = glob.glob("../*acled/settings/nigeria_acled.json")




summary_df_list = []

thresh_val_list = [0.3, 0.4, 0.5]
thresh_val_list = [0.5]
for thresh_val in thresh_val_list:
    # thresh_val = 0.4

    for json_path in json_path_list:
        batch_id = "_".join(json_path.split("/")[0].split("_")[:2])
        print(batch_id)


        # *****************
        # *****************
        # json_path = "settings/nigeria_acled.json"
        # json_path = "settings/settings_example.json"
        # json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
        # *****************
        # *****************


        s = Settings()
        s.load(json_path)

        predict_settings = s.data["predict"]["grid"]
        predict_hash = s.build_hash(predict_settings, nchar=7)

        predict_tag = s.config["predict_tag"]
        model_tag = s.config["model_tag"]
        surface_tag = s.config["surface_tag"]

        tasks = s.hashed_iter()


        print("-----")

        input_stage = s.data["surface"]["input_stage"]

        if input_stage == "s1":
            surface_list = [
                (0, s.data["surface"]["value_type"])
            ]
        else:
            raise ValueError("Surface input stage must be either `s1` or `s2`. (`{}` given)".format(input_stage))


        survey_path = "/sciclone/aiddata10/REU/projects/lab_oi_nigeria/data/sample/{}.csv".format(s.data["predict"]["acled"]["sample"])

        survey_df = pd.read_csv(survey_path)

        survey_df["binary"] = (survey_df["fatalities"] > 0).astype(int)

        if "lon" not in  survey_df.columns:
            survey_df["lon"] = survey_df["longitude"]


        if "lat" not in  survey_df.columns:
            survey_df["lat"] = survey_df["latitude"]

        print("Survey points: {}".format(len(survey_df)))


        for ix, (param_hash, _params) in enumerate(tasks):
            for surface_item in surface_list:
                print("Running: {} - {}".format(param_hash, surface_item))

                if input_stage == "s1":
                    _, input_name = surface_item
                    input_string = "_".join(str(i) for i in [
                        param_hash,
                        predict_hash,
                        s.config["version"],
                        predict_tag
                    ])

                surface_string = input_string + "_" + surface_tag
                s3_surface_path = os.path.join(s.base_path, "output/s3_surface/surface_{}_{}.tif".format(input_name, surface_string))
                print(s3_surface_path)

                # -------------------------------------

                cnn_surface_path = s3_surface_path

                validation_dir = os.path.dirname(os.path.dirname(cnn_surface_path)) + "/s4_validation"
                validation_fname = os.path.basename(cnn_surface_path)[:-4] + "_" + os.path.basename(survey_path)[:-4]
                validation_base = os.path.join(validation_dir, validation_fname)

                make_dir(validation_dir)

                cnn_surface_src = rasterio.open(cnn_surface_path, 'r')

                validation_data = []
                for i, row in survey_df.iterrows():
                    surface_data = {
                        "point": cnn_surface_src.sample([(row.lon, row.lat)]).next()[0],
                        "dim3": surface_win_mean(cnn_surface_src, row, 3),
                        "dim16": surface_win_mean(cnn_surface_src, row, 16),
                        "dim33": surface_win_mean(cnn_surface_src, row, 33),
                    }
                    for i in surface_data.keys():
                        try:
                            class_val = 1 if surface_data[i] > thresh_val else 0
                            class_val = -1 if class_val == 255 else class_val
                            surface_data[i+"_class"] = class_val
                        except:
                            surface_data[i+"_class"] = -1
                    validation_data.append(surface_data)

                validation_df = pd.DataFrame(validation_data)

                tmp_survey_df = survey_df.copy(deep=True)

                for i in validation_df.columns:
                    tmp_survey_df[i] = validation_df[i]

                tmp_survey_df = tmp_survey_df.loc[tmp_survey_df.point_class != -1]

                print("Original points: {}".format(len(survey_df)))
                print("Clean points: {}".format(len(tmp_survey_df)))

                metrics = ["tp", "fn", "tn", "fp", "accuracy", "precision", "recall", "f1"]

                tmp_summary_list = []
                # count = float(len(tmp_survey_df))
                for i in tmp_survey_df.columns:
                    if i.endswith("class"):
                        print(i)
                        y_true = tmp_survey_df["binary"]
                        y_pred = tmp_survey_df[i]
                        y_prob = tmp_survey_df[i[:-6]]
                        tmp_survey_df[i+"_match"] = (y_true == y_pred).astype(int)
                        tmp_survey_df[i+"_confusion"] =  None
                        tmp_survey_df.loc[((y_true == 1) & (y_pred == 1)), i+"_confusion"] = "tp"
                        tmp_survey_df.loc[((y_true == 0) & (y_pred == 0)), i+"_confusion"] = "tn"
                        tmp_survey_df.loc[((y_true == 0) & (y_pred == 1)), i+"_confusion"] = "fp"
                        tmp_survey_df.loc[((y_true == 1) & (y_pred == 0)), i+"_confusion"] = "fn"
                        tmp_summary = {}
                        tmp_summary["method"] = i
                        tp = sum((y_true == 1) & (y_pred == 1))
                        fn = sum((y_true == 1) & (y_pred == 0))
                        tn = sum((y_true == 0) & (y_pred == 0))
                        fp = sum((y_true == 0) & (y_pred == 1))
                        tmp_summary["tp"] = tp / float(tp+fn)
                        tmp_summary["fn"] = fn / float(fn+tp)
                        tmp_summary["tn"] = tn / float(tn+fp)
                        tmp_summary["fp"] = fp / float(fp+tn)
                        # tmp_summary["tp"] = sum((y_true == 1) & (y_pred == 1)) / count
                        # tmp_summary["fn"] = sum((y_true == 1) & (y_pred == 0)) / count
                        # tmp_summary["tn"] = sum((y_true == 0) & (y_pred == 0)) / count
                        # tmp_summary["fp"] = sum((y_true == 0) & (y_pred == 1)) / count
                        tmp_summary["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
                        tmp_summary["precision"] = sklearn.metrics.precision_score(y_true, y_pred)
                        tmp_summary["recall"] = sklearn.metrics.recall_score(y_true, y_pred)
                        tmp_summary["f1"] = sklearn.metrics.f1_score(y_true, y_pred)
                        tmp_summary_list.append(tmp_summary)
                        # ====================
                        # ====================
                        # ====================
                        # generate curves
                        auc = sklearn.metrics.roc_auc_score(y_true, y_prob)
                        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_prob)
                        # 1:1 line (noskill) data
                        ns_probs = [0 for _ in range(len(y_true))]
                        ns_auc = sklearn.metrics.roc_auc_score(y_true, ns_probs)
                        ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(y_true, ns_probs)
                        plt.figure()
                        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
                        plt.plot(fpr, tpr, marker='.', label='Actual')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.legend()
                        plt.title("ROC Curve")
                        plot_path = validation_base + "_roc_" + i + ".png"
                        plt.savefig(plot_path)
                        print('No Skill: ROC AUC=%.3f' % (ns_auc))
                        print('Actual: ROC AUC=%.3f' % (auc))
                        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_prob)
                        prc_precision, prc_recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_prob)
                        no_skill = len(y_true[y_true==1]) / len(y_true)
                        plt.figure()
                        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
                        plt.plot(prc_recall, prc_precision, marker='.', label='Actual')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.legend()
                        plt.title("PRC Curve")
                        plot_path = validation_base + "_prc_" + i + ".png"
                        plt.savefig(plot_path)
                        # ====================
                        # ====================
                        # ====================


                tmp_summary_df = pd.DataFrame(tmp_summary_list)

                tmp_summary_df["id"] = validation_fname
                tmp_summary_df["thresh"] = thresh_val

                tmp_summary_df = tmp_summary_df[["id", "thresh", "method"] + metrics]

                final_cols = ["lon", "lat", "fatalities", "binary"]
                for i in validation_df.columns:
                    final_cols.append(i)
                    if i.endswith("class"):
                        for j in ["match", "confusion"]:
                            final_cols.append(i+"_"+j)

                final_survey_df = tmp_survey_df[final_cols]


                final_survey_df.to_csv(validation_base + "_survey.csv", index=False)
                tmp_summary_df.to_csv(validation_base + "_summary.csv", index=False)
                summary_df_list.append(tmp_summary_df)
        break

final_summary_df = pd.concat(summary_df_list)
final_summary_path = os.path.join(validation_dir, "final_summary_{}.csv".format(timestamp))
print(final_summary_path)
final_summary_df.to_csv(final_summary_path, index=False)



"""
# check static hashes
from __future__ import print_function, division
from utils.settings_builder import Settings
from utils.data_prep import *
import glob

hashes = {}
for json_path in glob.glob("/home/userv/Desktop/laboi_batches_best/*.json"):
    s = Settings()
    s.load(json_path)
    static_hash = s.build_hash(s.static)
    hashes[static_hash] = {"imagery": s.static["sample_definition"]["acled"]["imagery"], "sample": s.static["sample_definition"]["acled"]["sample"]}
    print(static_hash, hashes[static_hash])
"""