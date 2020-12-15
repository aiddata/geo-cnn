"""
specify survey layer (with year) to validate

load s4_surface for same year (use json or fixed args to determine which surface?)

compare point (or buffer zs?) value of surface to survey value for each survey point

output dataframe of survey value, point value, buffer value along with percent differences? (include lat/lon to map errors)

"""


import os
import itertools
import errno
import glob
import time
import datetime
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.settings_builder import Settings


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class ConfusionMatrix():
    def __init__(self, true, pred, proba=False, thresh=0.5):
        self.true = true
        self.raw_pred = pred
        if proba:
            self.pred = (pred > thresh).astype(int)
        self.tp = sum((true == 1) & (self.pred == 1))
        self.fn = sum((true == 1) & (self.pred == 0))
        self.tn = sum((true == 0) & (self.pred == 0))
        self.fp = sum((true == 0) & (self.pred == 1))
        self.cm = (self.tp, self.fn, self.tn, self.fp)
        self.gen_rates()
        self.gen_performance_measures()
    def run(self):
        tpr, fnr, tnr, fpr = self.gen_rates()
        accuracy, precision, recall, f1 = self.gen_performance_measures()
        out = {
            "tp": tpr, "fn": fnr, "tn": tnr, "fp": fpr,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        }
        return out
    def gen_rates(self):
        tpr = self.calc_tp_rate()
        fnr = self.calc_fn_rate()
        tnr = self.calc_tn_rate()
        fpr = self.calc_fp_rate()
        return (tpr, fnr, tnr, fpr)
    def gen_performance_measures(self):
        accuracy = sklearn.metrics.accuracy_score(self.true, self.pred)
        precision = sklearn.metrics.precision_score(self.true, self.pred)
        recall = sklearn.metrics.recall_score(self.true, self.pred)
        f1 = sklearn.metrics.f1_score(self.true, self.pred)
        return (accuracy, precision, recall, f1)
    def calc_tp_rate(self):
        try:
            return self.tp / float(self.tp+self.fn)
        except:
            return None
    def calc_fn_rate(self):
        try:
            return self.fn / float(self.fn+self.tp)
        except:
            return None
    def calc_tn_rate(self):
        try:
            return self.tn / float(self.tn+self.fp)
        except:
            return None
    def calc_fp_rate(self):
        try:
            return self.fp / float(self.fp+self.tn)
        except:
            return None


timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime(
    '%Y_%m_%d_%H_%M_%S')

json_path_list = glob.glob("../*acled/settings/nigeria_acled.json")


# group versions of same temporal
version_groups = {}
for json_path in json_path_list:
    temporal = json_path.split("/")[1].split("_")[1]
    s = Settings()
    s.load(json_path)
    tag = "{}_{}".format(s.config["version"], s.config["predict_tag"])
    if temporal not in version_groups:
        version_groups[temporal] = []
    version_groups[temporal].append(json_path)




cm_list = []

thresh_val_list = [0.3, 0.35, 0.4, 0.45, 0.5]

for ix, temporal in enumerate(version_groups.keys()):
    # ==============
    # WARNING: THIS WILL NOT CURRENTLY WORK IF YOU HAVE MORE THAN ONE PARAM COMBO
    # it will just put all the params from same temporal group in same plot
    # ==============
    # init roc curve plot
    plt.figure(1)
    # init prc plot
    plt.figure(2)

    auc_scores = []
    for iy, json_path in enumerate(version_groups[temporal]):
    # for json_path in json_path_list:

        batch_id = "_".join(json_path.split("/")[1].split("_")[:2])
        print(batch_id)

        s = Settings()
        s.load(json_path)

        validation_dir = os.path.join(s.base_path, "output/s4_survey_validation")
        make_dir(validation_dir)

        predict_settings = s.data["predict"]["acled"]
        predict_hash = s.build_hash(predict_settings, nchar=7)

        version_tag = s.config["version"]
        predict_tag = s.config["predict_tag"]

        tasks = s.hashed_iter()

        for ix, (param_hash, _params) in enumerate(tasks):

            combo_temporal = "{}_{}".format(predict_settings["imagery"][0], predict_settings["sample"])
            print("Running: {} : {} - {} {}".format(param_hash, combo_temporal, version_tag, predict_tag))

            survey_prediction_path = os.path.join(s.base_path, "output/s1_predict/predict_{}_{}_{}_{}.csv".format(param_hash, predict_hash,  version_tag, predict_tag))
            survey_prediction_df = pd.read_csv(survey_prediction_path)
            survey_prediction_df = survey_prediction_df.loc[survey_prediction_df["group"] == "original"].copy(deep=True)
            survey_prediction_df = survey_prediction_df[[i for i in survey_prediction_df.columns if not i.startswith("feat_")]]
            survey_prediction_df['index'] = survey_prediction_df['data_id']
            survey_prediction_df.set_index('index', inplace=True)

            sample_path = os.path.join(s.base_path, "data/grid/sample_trim_{}_{}.csv".format(s.build_hash(s.static), version_tag))
            sample_df = pd.read_csv(sample_path)
            sample_df = sample_df.loc[sample_df["group"] == "original"].copy(deep=True)
            sample_df = sample_df[["data_id", "type", "drop"]]
            sample_df['index'] = sample_df['data_id']
            sample_df.set_index('index', inplace=True)

            survey_df = sample_df.join(survey_prediction_df, how="left", lsuffix="_sample", rsuffix="_predict")

            survey_df["binary"] = (survey_df["fatalities"] > 0).astype(int)

            if "lon" not in  survey_df.columns:
                survey_df["lon"] = survey_df["longitude"]


            if "lat" not in  survey_df.columns:
                survey_df["lat"] = survey_df["latitude"]

            print("Total Survey points: {}".format(len(survey_df)))

            survey_df = survey_df.loc[survey_df["type"] == "val"]
            print("Non-duplicate validation points: {}".format(len(survey_df)))

            y_true = survey_df["binary"]
            y_prob = survey_df["proba_1_1"]

            # ----------------------------------------------
            # generate confusion matrix at varying thresholds

            tmp_survey_df = survey_df.copy(deep=True)

            survey_df_list = []

            for thresh_val in thresh_val_list:

                validation_fname = "val_{}_{}_{}_{}".format(param_hash, predict_hash, version_tag, predict_tag)
                # validation_base = os.path.join(validation_dir, validation_fname)

                stats = ConfusionMatrix(y_true, y_prob, proba=True, thresh=thresh_val)

                tmp_cm_data = stats.run()
                cm_data = {}
                for k in tmp_cm_data.keys():
                    cm_data[k] = round(tmp_cm_data[k], 3)
                cm_data["id"] = validation_fname
                cm_data["temporal"] = combo_temporal
                cm_data["version"] = version_tag
                cm_data["thresh"] = thresh_val
                cm_list.append(cm_data)

                y_pred = stats.pred
                tmp_survey_df["thresh_{}_match".format(thresh_val)] = (y_true == y_pred).astype(int)
                col_name = "thresh_{}_confusion".format(thresh_val)
                tmp_survey_df[col_name] =  None
                tmp_survey_df.loc[((y_true == 1) & (y_pred == 1)), col_name] = "tp"
                tmp_survey_df.loc[((y_true == 0) & (y_pred == 0)), col_name] = "tn"
                tmp_survey_df.loc[((y_true == 0) & (y_pred == 1)), col_name] = "fp"
                tmp_survey_df.loc[((y_true == 1) & (y_pred == 0)), col_name] = "fn"
                survey_df_list.append(tmp_survey_df)


            final_survey_df = pd.concat(survey_df_list)
            final_survey_path = os.path.join(validation_dir, "{}_{}_{}_{}_{}_thresh-{}_survey.csv".format(timestamp, param_hash, predict_hash, version_tag, predict_tag, thresh_val))
            final_survey_df.to_csv(final_survey_path, index=False)

            # ----------------------------------------------
            # generate roc
            auc = sklearn.metrics.roc_auc_score(y_true, y_prob)
            auc_scores.append(auc)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_prob)
            # 1:1 line (noskill) data
            ns_probs = [0 for _ in range(len(y_true))]
            ns_auc = sklearn.metrics.roc_auc_score(y_true, ns_probs)
            ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(y_true, ns_probs)
            # plt.figure()
            plt.figure(1)
            # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            plt.plot(fpr, tpr, marker='.', label='Batch {} (AUC {})'.format(iy, round(auc,3)))
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.legend(loc='lower right')
            # plt.title("ROC Curve")
            # plot_path = os.path.join(validation_dir, "{}_{}_{}_{}_{}_roc.png".format(timestamp, param_hash, predict_hash, version_tag, predict_tag))
            # plt.savefig(plot_path)
            print('No Skill: ROC AUC=%.3f' % (ns_auc))
            print('Actual: ROC AUC=%.3f' % (auc))
            # ----------------------------------------------
            # generate prc
            prc_precision, prc_recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_prob)
            # no_skill = len(y_true[y_true==1]) / float(len(y_true))
            # plt.figure()
            plt.figure(2)
            plt.plot(prc_recall, prc_precision, marker='.', label='Batch {}'.format(iy))
            # plt.ylim(0.45, 1.05)
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.legend()
            # plt.title("PRC Curve")
            # plot_path = os.path.join(validation_dir, "{}_{}_{}_{}_{}_prc.png".format(timestamp, param_hash, predict_hash, version_tag, predict_tag))
            # plt.savefig(plot_path)


    if len(temporal) == 2:
        title_temporal = "201{} imagery to predict 201{} conflict".format(*temporal)

    elif len(temporal) == 4:
        title_temporal = "201{} h{} imagery to predict 201{} h{} conflict".format(*temporal)
    else:
        raise Exception("Invalid temporal string ({})".format(temporal))

    plt.figure(1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill (AUC 0.5)')
    plt.title("ROC Curve - Mean AUC: {}".format(round(np.mean(auc_scores), 3)))
    plt.suptitle("{}".format(title_temporal))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plot_path = os.path.join(validation_dir, "{}_{}_{}_{}_roc.png".format(timestamp, param_hash, predict_hash, temporal))
    plt.savefig(plot_path)
    plt.clf()

    plt.figure(2)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='No Skill')
    plt.title("Precision-Recall Curve")
    plt.suptitle("{}".format(title_temporal))
    plt.ylim(0.45, 1.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plot_path = os.path.join(validation_dir, "{}_{}_{}_{}_prc.png".format(timestamp, param_hash, predict_hash, temporal))
    plt.savefig(plot_path)
    plt.clf()

final_summary_df = pd.DataFrame(cm_list)
final_summary_df = final_summary_df[["id", "temporal", "version", "thresh", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]]
final_summary_path = os.path.join(validation_dir, "{}_summary.csv".format(timestamp))
final_summary_df.to_csv(final_summary_path, index=False)

# full confusion metrics only
full_cm_summary_df = final_summary_df[["temporal", "version", "thresh", "tp", "tn", "fp", "fn"]].copy(deep=True)
full_cm_summary_path = os.path.join(validation_dir, "{}_summary_full_cm_only.csv".format(timestamp))
full_cm_summary_df.to_csv(full_cm_summary_path, index=False)

# full performance metrics only
full_pm_summary_df = final_summary_df[["temporal", "version", "thresh", "accuracy", "precision", "recall", "f1"]].copy(deep=True)
full_pm_summary_path = os.path.join(validation_dir, "{}_summary_full_pm_only.csv".format(timestamp))
full_pm_summary_df.to_csv(full_pm_summary_path, index=False)


# temporal grouped confusion metrics only
temporal_cm_summary_df = final_summary_df[["temporal", "thresh", "tp", "tn", "fp", "fn"]].copy(deep=True)
temporal_cm_summary_df = temporal_cm_summary_df.groupby(["temporal", "thresh"]).agg({"tp": "mean", "tn": "mean", "fp": "mean", "fn": "mean"}).reset_index()
temporal_cm_summary_path = os.path.join(validation_dir, "{}_summary_temporal_cm_only.csv".format(timestamp))
temporal_cm_summary_df.to_csv(temporal_cm_summary_path, index=False)

# temporal grouped performance metrics only
temporal_pm_summary_df = final_summary_df[["temporal", "thresh", "accuracy", "precision", "recall", "f1"]].copy(deep=True)
temporal_pm_summary_df = temporal_pm_summary_df.groupby(["temporal", "thresh"]).agg({"accuracy": "mean", "precision": "mean", "recall": "mean", "f1": "mean"}).reset_index()
temporal_pm_summary_path = os.path.join(validation_dir, "{}_summary_temporal_pm_only.csv".format(timestamp))
temporal_pm_summary_df.to_csv(temporal_pm_summary_path, index=False)



# ===================================================================

# raise Exception("End of automated code")



# from nigeria_survey_validation import *


# final_summary_df.groupby("thresh").agg({"tp":"mean", "fn":"mean", "fp":"mean", "tn": "mean"})
# final_summary_df.groupby("thresh").agg({"accuracy":"mean", "precision":"mean", "recall":"mean", "f1": "mean"})




# w = final_summary_df.copy(deep=True)
# w["version"] = w.apply(lambda x: x["id"].split("_")[-2] , axis=1)

# w.groupby("version").agg({"accuracy":"mean", "precision":"mean", "recall":"mean"})




# y = w.loc[w["thresh"] == 0.35].copy(deep=True)
# z = y[["id", "temporal", "accuracy", "precision", "recall"]].copy(deep=True)

# z.groupby("temporal").agg({"accuracy":"mean", "precision":"mean", "recall":"mean"})



# ===================================================================


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