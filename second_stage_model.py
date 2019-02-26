

import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np

import matplotlib.pyplot as plot


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

lsms_out_path = os.path.join(base_path, "output/predict_28dab38_2019_02_21_11_57_19.csv")

lsms_out = pd.read_csv(lsms_out_path, quotechar='\"',
                       na_values='', keep_default_na=False,
                       encoding='utf-8')

def quick(x,y, model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)
    model.fit(x_train, y_train)
    model_preds = model.predict(x_test)
    r2 = r2_score(y_test, model_preds)
    return r2


test_feat_labels = ["feat_{}".format(i) for i in xrange(1,513)]



# plot.hist(lsms_out['ntl_2010'], bins=max(lsms_out['ntl_2010']), alpha=0.5, histtype='bar', ec='black')
# plot.xlabel('NTL')
# plot.ylabel('Frequency')
# plot.title('Histogram of NTL Values')
# plot.show()


# plot.hist(np.array([lsms_out[i] for i in test_feat_labels]).flatten())
# # plot.yscale("log")
# plot.xlabel('Features')
# plot.ylabel('Frequency')
# plot.title('Histogram of All Features Values')
# plot.show()

print "NTL (Linear)"
quick(lsms_out[['ntl_2010']], lsms_out["cons"], model=LinearRegression())

print "Features (Linear)"
quick(lsms_out[test_feat_labels], lsms_out["cons"], model=LinearRegression())

print "NTL (Linear)"
quick(lsms_out[['ntl_2010']], lsms_out["cons"], model=Ridge())

print "Features (Linear)"
quick(lsms_out[test_feat_labels], lsms_out["cons"], model=Ridge())

