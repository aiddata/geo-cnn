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


# *****************
# *****************
json_path = "settings/nigeria_acled.json"
# json_path = "settings/settings_example.json"
# json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


s = Settings()
s.load(json_path)

predict_settings = s.data[s.config["predict"]]
predict_hash = s.build_hash(predict_settings, nchar=7)

tasks = s.hashed_iter()


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class ConfusionMatrix():
    def __init__(self, true, pred):
        self.true = true
        self.pred = pred
        self.tp = sum((true == 1) & (pred == 1))
        self.fn = sum((true == 1) & (pred == 0))
        self.tn = sum((true == 0) & (pred == 0))
        self.fp = sum((true == 0) & (pred == 1))
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


for ix, (param_hash, params) in enumerate(tasks):

    print "Running: {} - ".format(param_hash)

    input_string = "_".join(str(i) for i in [
        param_hash,
        predict_hash,
        s.config["version"],
        s.config["predict_tag"]
    ])

    # predict_25212b1_da9a7de_v20a_p20b

    s1_predict_path = os.path.join(s.base_path, "output/s1_predict/predict_{}.csv".format(input_string))

    s1_predict_df = pd.read_csv(s1_predict_path)

    y_true = s1_predict_df["pred_yval"]
    y_pred = s1_predict_df["pred_class"]
    y_prob = s1_predict_df["proba_1_1"]

    stats = ConfusionMatrix(y_true, y_pred)
    summary = stats.run()
    summary_df = pd.DataFrame([summary])

    # -----
    validation_dir = os.path.dirname(os.path.dirname(s1_predict_path)) + "/s1_validation"

    validation_fname = "validation_{}".format(input_string)
    validation_base = os.path.join(validation_dir, validation_fname)

    make_dir(validation_dir)
    # -----

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
    plot_path = validation_base + "_roc.png"
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
    plot_path = validation_base + "_prc.png"
    plt.savefig(plot_path)
    # f1 = sklearn.metrics.f1_score(y_true, yhat)
    # auc = auc(recall, precision)
    # prc_f1, prc_auc = sklearn.metrics.f1_score(y_true, yhat), auc(prc_recall, prc_precision)
    # print('Actual: f1=%.3f auc=%.3f' % (prc_f1, prc_auc))

    metrics = ["tp", "fn", "tn", "fp", "accuracy", "precision", "recall", "f1"]
    summary_df = summary_df[metrics]
    summary_df.to_csv(validation_base + "_summary.csv", index=False)
    summary_df





