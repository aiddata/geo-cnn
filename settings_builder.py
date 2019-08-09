import os
import json
import hashlib
import itertools
import warnings
import pprint
import pandas as pd
import numpy as np


class Settings():


    def __init__(self):
        self.base_path = None

        self.data = None
        self.config = None
        self.static = None

        self.param_dicts = None
        self.param_count = None


    def json_sha1_hash(self, hash_obj):
        hash_json = json.dumps(hash_obj,
                            sort_keys = True,
                            ensure_ascii = True,
                            separators=(', ', ': '))
        hash_builder = hashlib.sha1()
        hash_builder.update(hash_json)
        hash_sha1 = hash_builder.hexdigest()
        return hash_sha1


    def build_hash(self, hash_obj, nchar=7):
        if not isinstance(nchar, int):
            raise ValueError("Settings: invalid nchar value given ({})".format(nchar))
        if nchar < 1:
            nchar = -1
            warnings.warn("Settings: character limit for hash given ({}) is less than one. Using full hash.".format(nchar))
        tmp_hash = self.json_sha1_hash(hash_obj)
        if nchar > 0:
            tmp_hash = tmp_hash[:nchar]
        return tmp_hash


    def gen_dict_product(self, d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))


    def dict_product(self, d):
        return list(self.gen_dict_product(d))


    def _param_dict_check(self):
        if self.param_dicts is None:
            raise Exception("Settings: No parameter data loaded")


    def set_param_count(self):
        self._param_dict_check()
        self.param_count = len(self.param_dicts)


    def get_param_count(self):
        self.set_param_count()
        return self.param_count


    def check_static_params(self):
        self._param_dict_check()
        static_match = None
        for i in self.param_dicts:
            if "static" not in i:
                raise ValueError("Settings: no static params found in param dict")
            if static_match is None:
                static_match = i["static"]
            if static_match != i["static"]:
                raise Exception("Settings: static params do not match")


    def load_csv(self, data):
        """csv should contain column `field` which has the
        format `hash`_`version` for each row
        """
        self._csv_check(data)

        path = data["csv"]["path"]
        field = data["csv"]["field"]

        if not os.path.isfile(path):
            raise Exception("Settings: CSV file path provided does not exist ({})".format(path))

        id_df = pd.read_csv(path, sep=",", encoding='utf-8')
        id_list = id_df[field]

        self.param_dicts = []

        param_json_format = os.path.join(self.base_path, "output/s1_params/params_{}.json")
        for id_str in id_list:
            param_json_path = param_json_format.format(id_str)
            with open(param_json_path) as j:
                self.param_dicts.append(json.load(j))
        self.check_static_params()
        self.static = self.param_dicts[0]["static"]


    # def load_single(self, param_dict):
    #     self.param_dicts = [param_dict]
    #     self.static = param_dict["static"]
    #     self.base_path = self.static["base_path"]
    #     print("\nPreparing single parameter set:")
    #     pprint.pprint(param_dict, indent=4)


    def load_batch(self, data):
        self._batch_check(data)
        self.static = data["static"]
        self.param_dicts = self.dict_product(data["batch"])
        for p in self.param_dicts:
            p.update({"static": self.static})
        # self.param_count = np.prod([len(i) for i in data.values()])
        print("\nPreparing batch parameter set:")
        pprint.pprint(data, indent=4)
        print("\n")


    def _batch_check(self, data):
        if not isinstance(data, dict):
            raise ValueError("Settings: invalid JSON data provided (type: {})".format(type(data)))
        if not "batch" in data:
            raise ValueError("Settings: no batch params found")
        if not isinstance(data["batch"], dict):
            raise ValueError("Settings: invalid batch params format (type: {})".format(type(data["batch"])))
        for k in data["batch"]:
            if not isinstance(data["batch"][k], list):
                raise ValueError("Settings: batch params must be lists ({})".format(k))


    def _csv_check(self, data):
        if not isinstance(data, dict):
            raise ValueError("Settings: invalid JSON data provided (type: {})".format(type(data)))
        if not "csv" in data:
            raise ValueError("Settings: no csv params found")
        if not isinstance(data["csv"], dict):
            raise ValueError("Settings: invalid csv params format (type: {})".format(type(data["csv"])))


    def _general_check(self, data):
        if not isinstance(data, dict):
            raise ValueError("Settings: invalid JSON data provided (type: {})".format(type(data)))
        if not "static" in data:
            raise ValueError("Settings: no static params found")
        if not "config" in data:
            raise ValueError("Settings: no config params found")
        if not isinstance(data["static"], dict):
            raise ValueError("Settings: invalid static params format (type: {})".format(type(data["static"])))
        if not isinstance(data["config"], dict):
            raise ValueError("Settings: invalid config params format (type: {})".format(type(data["config"])))


    def load_json(self, arg):
        if isinstance(arg, str):
            try:
                data = json.loads(arg)
            except:
                if os.path.isfile(arg):
                    try:
                        f = open(arg, 'r')
                        data = json.load(f)
                        f.close()
                    except:
                        raise Exception("Settings: load_json given string that is not serialized JSON or valid JSON file path")
                else:
                    raise Exception("Settings: load_json given string that is not serialized JSON or existing file path")
        else:
            data = arg
        self._general_check(data)
        self.data = data
        self.config = data["config"]
        self.base_path = self.config["base_path"]
        if self.config["mode"] == "batch":
            self.load_batch(data)
        elif self.config["mode"] == "csv":
            self.load_csv(data)

        else:
            raise ValueError("Settings: invalid mode provided ({})".format(self.config["mode"]))


    def load(self, arg):
        self.load_json(arg)


    def get_hash_list(self, nchar=7):
        self._param_dict_check()
        hash_list = [self.build_hash(pdict, nchar=nchar) for pdict in self.param_dicts]
        return hash_list


    def gen_hashed_iter(self, nchar=7):
        self._param_dict_check()
        for pdict in self.param_dicts:
            yield self.build_hash(pdict, nchar=nchar), pdict


    def hashed_iter(self, nchar=7):
        return list(self.gen_hashed_iter(nchar=nchar))


    def write_to_json(self, phash, pdict):
        output_format = os.path.join(self.base_path, "output/s1_train/train_{}_{}.json")
        path = output_format.format(phash, self.config["version"])
        with open(path, "w", 0) as f:
            json.dump(pdict, f)


    def save_params(self):
        self._param_dict_check()
        output_format = os.path.join(self.base_path, "output/s1_params/params_{}_{}.json")
        for phash, pdict in self.gen_hashed_iter():
            path = output_format.format(phash, self.config["version"])
            with open(path, "w", 0) as f:
                json.dump(pdict, f, indent=4, sort_keys=True)
