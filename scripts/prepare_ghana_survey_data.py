

import os
import glob
import pandas as pd
import geopandas as gpd


# base = "/sciclone/aiddata10/REU/projects"
base = "/home/userz/Desktop"
project = "mcc_ghana"

survey_data = os.path.join(base, project, "data/surveys")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# DHS 2008

dhs_base = [
    i for i in glob.glob(
        os.path.join(survey_data, "DHS", "GH_2008_DHS_*"))
    if os.path.isdir(i)
][0]

dhs_data_path = os.path.join(dhs_base, "GHHR5ADT/GHHR5AFL.DTA")

# geographic data (shapefile)
dhs_coords_path = os.path.join(dhs_base, "GHGE5AFL/GHGE5AFL.shp")

dhs_data = pd.read_stata(dhs_data_path)
dhs_coords = gpd.read_file(dhs_coords_path)


var_list = ["hhid", "hv001", "hv005", "hv271"]
name_list = ["hhid", "cluster", "weight", "wealthscore"]

dhs_data = dhs_data[var_list]
dhs_data.columns = name_list


var_list = ["DHSCLUST", "LATNUM", "LONGNUM"]
name_list = ["cluster", "lat", "lon"]

dhs_coords = dhs_coords[var_list]
dhs_coords.columns = name_list

dhs_merge = dhs_data.merge(dhs_coords, on="cluster")

dhs_geo = dhs_merge.copy(deep=True)

# drop lat or lon = 0 or NA / empty
dhs_geo = dhs_geo.loc[~dhs_geo["lat"].isnull() | ~dhs_geo["lon"].isnull()]
dhs_geo = dhs_geo.loc[(dhs_geo["lat"] != 0) & (dhs_geo["lon"] != 0)]

dhs_final = dhs_geo

# output household
dhs_household = dhs_final.copy(deep=True)
dhs_household_path = os.path.join(survey_data, "final/ghana_2008_dhs_household.csv")
dhs_household.to_csv(dhs_household_path, index=False, encoding='utf-8')

# cluster
#   group by lat/lon
#   mean wealthscore, mean ntl, number of households n in group

dhs_precluster = dhs_final.copy(deep=True)
dhs_precluster["lonlat"] = dhs_precluster.apply(
    lambda z: "{0}_{1}".format(z["lon"], z["lat"]), axis=1)
dhs_precluster["n"] = 1

agg_fields = {
    "lon": "last",
    "lat": "last",
    "wealthscore": "mean",
    "n": "sum"
}

dhs_cluster = dhs_precluster.groupby(["lonlat"]).agg(agg_fields).reset_index()

# output cluster
dhs_cluster_path = os.path.join(survey_data, "final/ghana_2008_dhs_cluster.csv")
dhs_cluster.to_csv(dhs_cluster_path, index=False, encoding='utf-8')


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# DHS 2014

dhs_base = [
    i for i in glob.glob(
        os.path.join(survey_data, "DHS", "GH_2014_DHS_*"))
    if os.path.isdir(i)
][0]

dhs_data_path = os.path.join(dhs_base, "GHHR72DT/GHHR72FL.DTA")
dhs_coords_path = os.path.join(dhs_base, "GHGE71FL/GHGE71FL.shp")

dhs_data = pd.read_stata(dhs_data_path)
dhs_coords = gpd.read_file(dhs_coords_path)


var_list = ["hhid", "hv001", "hv005", "hv271"]
name_list = ["hhid", "cluster", "weight", "wealthscore"]

dhs_data = dhs_data[var_list]
dhs_data.columns = name_list


var_list = ["DHSCLUST", "LATNUM", "LONGNUM"]
name_list = ["cluster", "lat", "lon"]

dhs_coords = dhs_coords[var_list]
dhs_coords.columns = name_list

dhs_merge = dhs_data.merge(dhs_coords, on="cluster")

dhs_geo = dhs_merge.copy(deep=True)

# drop lat or lon = 0 or NA / empty
dhs_geo = dhs_geo.loc[~dhs_geo["lat"].isnull() | ~dhs_geo["lon"].isnull()]
dhs_geo = dhs_geo.loc[(dhs_geo["lat"] != 0) & (dhs_geo["lon"] != 0)]

dhs_final = dhs_geo

# output household
dhs_household = dhs_final.copy(deep=True)
dhs_household_path = os.path.join(survey_data, "final/ghana_2014_dhs_household.csv")
dhs_household.to_csv(dhs_household_path, index=False, encoding='utf-8')

# cluster
#   group by lat/lon
#   mean wealthscore, number of households n in group

dhs_precluster = dhs_final.copy(deep=True)
dhs_precluster["lonlat"] = dhs_precluster.apply(
    lambda z: "{0}_{1}".format(z["lon"], z["lat"]), axis=1)
dhs_precluster["n"] = 1

agg_fields = {
    "lon": "last",
    "lat": "last",
    "wealthscore": "mean",
    "n": "sum"
}

dhs_cluster = dhs_precluster.groupby(["lonlat"]).agg(agg_fields).reset_index()

# output cluster
dhs_cluster_path = os.path.join(survey_data, "final/ghana_2014_dhs_cluster.csv")
dhs_cluster.to_csv(dhs_cluster_path, index=False, encoding='utf-8')
