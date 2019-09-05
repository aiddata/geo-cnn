
import numpy as np
import pandas as pd

from nearest_neighbor import NN


def afb_q52a(data):

    afb_path = "/sciclone/aiddata10/REU/projects/lab_oi_nigeria/data/afb_data/r6/afb_full_r6_nig.csv"
    afb = pd.read_csv(afb_path, sep=",", encoding='utf-8')
    afb_field = "q52a"
    afb = afb[[afb_field, "longitude", "latitude"]]
    afb = afb.loc[(afb[afb_field] >=0) & (afb[afb_field] < 9)].copy(deep=True)

    nn = NN(afb, k=1, agg_func=None)
    nn.build_tree()

    df = data.copy(deep=True)

    print nn.query((0,0), afb_field)
    df["afb_q52a"] = df.apply(lambda x: nn.query((x.lon, x.lat), afb_field), axis=1)

    return np.array([df["afb_q52a"].values]).T
