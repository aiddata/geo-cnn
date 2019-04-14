import os
import json
import hashlib
import itertools
import warnings
import pprint
import pandas as pd
import numpy as np


class Settings():


    def __init__(self, base, quiet=False):
        self.base_path = base
        self.param_dicts = None
        self.static = None
        self.param_count = None
        self.quiet = quiet


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
        if not "static" in data:
            raise ValueError("Settings: no static params found")
        if not isinstance(data["static"], dict):
            raise ValueError("Settings: invalid static params format (type: {})".format(type(data["static"])))


    def load_batch(self, pranges):
        self._batch_check(pranges)
        self.static = pranges["static"]
        self.param_dicts = [i for i in self.dict_product(pranges["batch"])]
        # self.param_dicts = [i.update({"static": self.static}) for i in self.dict_product(pranges["batch"])]
        # self.param_count = np.prod([len(i) for i in pranges.values()])
        if not self.quiet:
            print("\nPreparing batch parameter set:")
            pprint.pprint(pranges, indent=4)


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
        self._batch_check(data)
        self.load_batch(data)


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


    def load_csv(self, path, field="hash"):
        if not os.path.isfile(file):
            raise Exception("Settings: CSV file path provided does not exist ({})".format(path))

        hash_df = pd.read_csv(path, sep=",", encoding='utf-8')
        hash_list = hash_df[field]

        self.param_dicts = []

        param_json_format = os.path.join(self.base_path, "output/s1_params/params_{}.json")
        for param_hash in hash_list:
            param_json_path = param_json_format.format(param_hash)
            with open(param_json_path) as j:
                self.param_dicts.append(json.load(j))
        self.check_static_params()
        self.static = self.param_dicts[0]["static"]


    def load_single(self, param_dict):
        self.param_dicts = [param_dict]
        self.static = param_dict["static"]
        if not self.quiet:
            print("\nPreparing single parameter set:")
            pprint.pprint(param_dict, indent=4)


    def load(self, arg):
        if isinstance(arg, str) and arg.endswith("csv"):
            self.load_csv(arg)
        else:
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
        output_format = os.path.join(self.base_path, "output/s1_params/params_{}.json")
        path = output_format.format(phash)
        with open(path, "w", 0) as f:
            json.dump(pdict, f)
