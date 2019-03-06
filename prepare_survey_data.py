

import os
import errno
import glob
import functools
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterstats as rs
from shapely.geometry import Point


base = "/sciclone/aiddata10/REU/projects"
project = "mcc_tanzania"

survey_data = os.path.join(base, project, "data/survey")


# --------------------------------------------------------------------------


dhs_base = [
    i for i in glob.glob(
        os.path.join(survey_data, "DHS", "TZ_2010_DHS_*"))
    if os.path.isdir(i)
][0]

dhs_data_path = os.path.join(dhs_base, "TZHR63DT/TZHR63FL.DTA")
dhs_coords_path = os.path.join(dhs_base, "TZGE61FL/TZGE61FL.shp")

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


# add ntl extract
# drop lat or lon = 0 or NA / empty
# buffer 5km and get mean extract

dhs_geo = dhs_merge.copy(deep=True)
dhs_geo = dhs_geo.loc[~dhs_geo["lat"].isnull() | ~dhs_geo["lon"].isnull()]
dhs_geo = dhs_geo.loc[(dhs_geo["lat"] != 0) & (dhs_geo["lon"] != 0)]
dhs_buffer = 5000/111123.0
dhs_geo['geometry'] = dhs_geo.apply(lambda z: Point(z['lon'], z['lat']), axis=1)
dhs_geo['geometry'] = dhs_geo.apply(lambda z: z['geometry'].buffer(dhs_buffer), axis=1)
dhs_geo = gpd.GeoDataFrame(dhs_geo)

ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"

ntl_years = [2010]

dhs_final = 0
for y in ntl_years:
    ntl_path = glob.glob(os.path.join(ntl_base, "*{0}*.tif".format(y)))[0]
    extract_geojson = rs.zonal_stats(dhs_geo, ntl_path, stats="mean", geojson_out=True)
    extract_list = [i['properties'] for i in extract_geojson]
    dhs_extract = pd.DataFrame(extract_list)
    dhs_extract.columns = ["ntl_{0}".format(y) if i == "mean" else i for i in dhs_extract.columns]
    if not dhs_final:
        dhs_final = dhs_extract.copy(deep=True)
    else:
        dhs_extract = dhs_extract[["cluster", "ntl_{0}".format(y)]]
        dhs_final = dhs_final.merge(dhs_extract, on="cluster")


# output household
dhs_household = dhs_final.copy(deep=True)
dhs_household_path = os.path.join(survey_data, "final/tanzania_dhs_household.csv")
dhs_household.to_csv(dhs_household_path, index=False, encoding='utf-8')



# cluster
# group by lat/lon
# mean wealthscore, mean ntl, number of households n in group

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

for y in ntl_years: agg_fields["ntl_{0}".format(y)] = "mean"

dhs_cluster = dhs_precluster.groupby(["lonlat"]).agg(agg_fields).reset_index()


# output cluster
dhs_cluster_path = os.path.join(survey_data, "final/tanzania_dhs_cluster.csv")
dhs_cluster.to_csv(dhs_cluster_path, index=False, encoding='utf-8')



# --------------------------------------------------------------------------


lsms_base = [
    i for i in glob.glob(
        os.path.join(survey_data, "LSMS", "TZA_2012_*"))
    if os.path.isdir(i)
][0]

lsms_data_1_path = os.path.join(lsms_base, "ConsumptionNPS3.dta")
lsms_data_2_path = os.path.join(lsms_base, "HouseholdGeovars_Y3.dta")
lsms_data_3_path = os.path.join(lsms_base, "HH_SEC_A.dta")
lsms_data_4_path = os.path.join(lsms_base, "HH_SEC_I.dta")

lsms_data_1 = pd.read_stata(lsms_data_1_path)
lsms_data_2 = pd.read_stata(lsms_data_2_path)
lsms_data_3 = pd.read_stata(lsms_data_3_path)
lsms_data_4 = pd.read_stata(lsms_data_4_path)



lsms_data_1["cons"] = lsms_data_1["expmR"] / (365 * lsms_data_1["adulteq"])
lsms_data_1["cons"] = lsms_data_1["cons"] * 112.69 / (585.52 * np.mean([130.72, 141.01]))
lsms_data_1 = lsms_data_1[["y3_hhid", "cons"]]
lsms_data_1.columns = ["hhid", "cons"]


lsms_data_2 = lsms_data_2[["y3_hhid", "lat_dd_mod", "lon_dd_mod"]]
lsms_data_2.columns = ["hhid", "lat", "lon"]


lsms_data_3 = lsms_data_3[["y3_hhid", "y3_rural", "y3_weight"]]
lsms_data_3.columns = ["hhid", "rururb", "weight"]


lsms_data_4a = lsms_data_4[["y3_hhid", "hh_i07_1"]]
lsms_data_4a.columns = ["hhid", "room"]
# drop NA rows
lsms_data_4a = lsms_data_4a.loc[~lsms_data_4a["hhid"].isnull() & ~lsms_data_4a["room"].isnull()]


lsms_data_4b = lsms_data_4[["y3_hhid", "hh_i09"]]
lsms_data_4b.columns = ["hhid", "metal"]
lsms_data_4b = lsms_data_4b.loc[lsms_data_4b["metal"] == "METAL SHEETS (GCI)"]


# merge
lsms_merge_list = [lsms_data_1, lsms_data_2, lsms_data_3, lsms_data_4a, lsms_data_4b]
lsms_merge = functools.reduce(lambda l,r: pd.merge(l, r, on="hhid"), lsms_merge_list)



lsms_geo = lsms_merge.copy(deep=True)
lsms_geo = lsms_geo.loc[~lsms_geo["lat"].isnull() | ~lsms_geo["lon"].isnull()]
lsms_geo = lsms_geo.loc[(lsms_geo["lat"] != 0) & (lsms_geo["lon"] != 0)]
lsms_buffer = 5000/111123.0
lsms_geo['geometry'] = lsms_geo.apply(lambda z: Point(z['lon'], z['lat']), axis=1)
lsms_geo['geometry'] = lsms_geo.apply(lambda z: z['geometry'].buffer(lsms_buffer), axis=1)
lsms_geo = gpd.GeoDataFrame(lsms_geo)

ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"

ntl_years = [2010]

lsms_final = 0
for y in ntl_years:
    ntl_path = glob.glob(os.path.join(ntl_base, "*{0}*.tif".format(y)))[0]
    extract_geojson = rs.zonal_stats(lsms_geo, ntl_path, stats="mean", geojson_out=True)
    extract_list = [i['properties'] for i in extract_geojson]
    lsms_extract = pd.DataFrame(extract_list)
    lsms_extract.columns = ["ntl_{0}".format(y) if i == "mean" else i for i in lsms_extract.columns]
    if not lsms_final:
        lsms_final = lsms_extract.copy(deep=True)
    else:
        lsms_extract = lsms_extract[["cluster", "ntl_{0}".format(y)]]
        lsms_final = lsms_final.merge(lsms_extract, on="cluster")


# output household
lsms_household = lsms_final.copy(deep=True)
lsms_household_path = os.path.join(survey_data, "final/tanzania_lsms_household.csv")
lsms_household.to_csv(lsms_household_path, index=False, encoding='utf-8')



# cluster
# group by lat/lon
# mean wealthscore, mean ntl, number of households n in group

lsms_precluster = lsms_final.copy(deep=True)
lsms_precluster["lonlat"] = lsms_precluster.apply(
    lambda z: "{0}_{1}".format(z["lon"], z["lat"]), axis=1)
lsms_precluster["n"] = 1

agg_fields = {
    "lon": "last",
    "lat": "last",
    "cons": "mean",
    "n": "sum"
}

for y in ntl_years: agg_fields["ntl_{0}".format(y)] = "mean"

lsms_cluster = lsms_precluster.groupby(["lonlat"]).agg(agg_fields).reset_index()


# output cluster
lsms_cluster_path = os.path.join(survey_data, "final/tanzania_lsms_cluster.csv")
lsms_cluster.to_csv(lsms_cluster_path, index=False, encoding='utf-8')




# --------------------------------------------------------------------------



def retrieve_and_save(df, fns, out_dir, names, keys):
    df = df[(df.lat!=0) & (df.lon!=0)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for name, key in zip(names, keys):
        np.save(os.path.join(out_dir, name), df[key])


# dhs
out_dir = os.path.join(survey_data, "final/dhs")
names = ['lats', 'lons', 'assets', 'nightlights', 'households']
keys = ['lat', 'lon', 'wealthscore', 'ntl_2010', 'n']
retrieve_and_save(dhs_cluster, dhs_cluster_path, out_dir, names, keys)

# lsms
out_dir = os.path.join(survey_data, "final/lsms")
names = ['lats', 'lons', 'consumptions', 'nightlights', 'households']
keys = ['lat', 'lon', 'cons', 'ntl_2010', 'n']
retrieve_and_save(lsms_cluster, lsms_cluster_path, out_dir, names, keys)
