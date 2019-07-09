

import os

from settings_builder import Settings


# *****************
# *****************
json_path = "settings/settings_example.json"
json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_path)
# *****************
# *****************


s = Settings()
s.load(json_path)


predict_settings = s.data[s.config["predict"]]
predict_hash = s.build_hash(predict_settings, nchar=7)


s3_info = s.data["third_stage"]
model_list = s3_info["predict"]["models"]

model_tag = s.config["model_tag"]
surface_tag = s.config["surface_tag"]


tasks = s.hashed_iter()

qlist = []

for ix, (param_hash, params) in enumerate(tasks):

    for model_name in model_list:

        s3_s2_string = "_".join(str(i) for i in [
            model_name,
            param_hash,
            predict_hash,
            s3_info["grid"]["boundary_id"],
            s3_info["predict"]["imagery_year"],
            s.config["version"],
            s.config["predict_tag"],
            s.config["model_tag"]
        ])

        s3_s2_path = os.path.join(s.base_path, "output/s3_s2_predict/predict_{}.csv".format(s3_s2_string))

        s4_path = os.path.join(s.base_path, "output/s4_surface/surface_{}_{}.csv".format(s3_s2_string, surface_tag))
