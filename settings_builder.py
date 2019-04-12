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


    def dict_product(self, d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            return dict(zip(keys, element))


    def _param_dict_check(self):
        if self.param_dicts is None:
            raise Exception("Settings: No parameter data loaded")

    def set_param_count(self):
        self._param_dict_check()
        self.param_count = len(self.param_dicts)


    def get_param_count(self):
        self.set_param_count()
        return self.param_count


    def load_csv(self, path, field="hash"):
        hash_df = pd.read_csv(path, sep=",", encoding='utf-8')
        hash_list = hash_df[field]

        self.param_dicts = []

        param_json_format = os.path.join(self.base_path, "output/s1_params/params_{}.json")
        for param_hash in hash_list:
            param_json_path = param_json_format.format(param_hash)
            with open(param_json_path) as j:
                self.param_dicts.append(json.load(j))


    def load_batch(self, pranges):
        for i in pranges:
            if not isinstance(pranges[i], list):
                raise ValueError("Settings: invalid batch format (`{}` not given as list)".format(i))
        self.param_dicts = self.dict_product(pranges)
        # self.param_count = np.prod([len(i) for i in pranges.values()])
        if not self.quiet:
            print("\nPreparing parameter set:")
            pprint.pprint(pranges, indent=4)


    def load_single(self, param_dict):
        self.param_dicts = [param_dict]


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
        if isinstance(data, dict):
            batch = True
            for i in data:
                if not isinstance(data[i], list):
                    batch = False
                    self.load_single(data)
                    break
            if batch:
                self.load_batch(data)
        else:
            raise ValueError("Settings: invalid JSON data provided (type: {})".format(type(data)))


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
