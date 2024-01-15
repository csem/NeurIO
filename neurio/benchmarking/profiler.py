#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 05.10.23
Description:  Profiler
"""
import json
from typing import List, Dict, Any
import copy

VERSION = "0.0.1"

base_skeleton_dictionnary = {
    "profiler_version": VERSION,
    "model": {
        "model_name": None,
        "framework": None,
        "model_datetime": None,
        "compile_datetime": None,
        "inputs": None,
        "outputs": None,
        "parameters": None,
        "macc": None,
    },
    "device": {
        "name": None,
        "port": None,
        "dev_type": None,
        "desc": None,
        "dev_id": None,
        "system": None,
        "sys_clock": None,
        "bus_clock": None,
        "attrs": []
    },
    "runtime": {
        "name": None,
        "version": [],
        "tools_version": {
        }
    },
    "inference": {
        "batch_size": None,
        "model_load_time": None,
        "preprocess_time": [],
        "inference_time": [],
    },
}


class Profiler:
    """
    Profiler class
    """
    def __init__(self, json_data: Dict = None):
        if json_data is None:
            self.json_data = copy.deepcopy(dict(base_skeleton_dictionnary))
        else:
            self.json_data = json_data

    @staticmethod
    def merge(other: [List, Any]):

        def merge_two_dicts(x, y):
            if x is None and y is None:
                return None
            elif x is None and y:
                return y
            elif y is None and x:
                return x

            else: # case values in each dict
                if isinstance(x, list) or isinstance(y, list):
                    if x is None or (isinstance(x, list) and len(x)==0):
                        return y
                    elif y is None or (isinstance(y, list) and len(y)==0):
                        return x
                    else:
                        if isinstance(x, list) and not isinstance(y, list):
                            return x+[y]
                        elif isinstance(y, list) and not isinstance(x, list):
                            return y+[x]
                        else:
                            return x+y

                elif isinstance(x, dict) or isinstance(y, dict):
                    if x is None or len(x) == 0:
                        return y
                    elif y is None or len(y) == 0:
                        return x
                    else:
                        res = {}
                        for k in list(x.keys()):
                            if k in list(y.keys()):
                                res[k] = merge_two_dicts(x[k], y[k])
                        return res
                else:
                    if x == y:
                        return x
                    else:
                        return [x, y]

        def merge_two_profilers(x, y):
            if isinstance(x, dict):
                x = x.copy()
            else:
                x = x.__to_dict__().copy()

            if isinstance(y, dict):
                y = y.copy()
            else:
                y = y.__to_dict__().copy()

            res = merge_two_dicts(x, y)

            new_p = Profiler(res)
            return new_p

        if not isinstance(other, list):
            other = [other]
        profiler = other[0]
        for p in other[1:]:
            profiler = merge_two_profilers(profiler, p)

        return profiler

    def __to_dict__(self):
        return dict(self.json_data)

    def __setitem__(self, key, value):
        if key not in self.json_data:
            raise KeyError("Key not found: " + str(key))
        self.json_data[key] = value
        #setattr(self, key, value)

    def __getitem__(self, key):
        if key not in self.json_data:
            raise KeyError("Key not found: " + str(key))
        return self.json_data[key]

    def __str__(self):
        return str(self.json_data)
""" 
def __setitem__(self, key, value):
    if key not in self.json_data:
        raise KeyError("Key not found: " + str(key))
    self.json_data[key] = value
    #setattr(self, key, value)

def __getitem__(self, key):
    if key not in self.json_data:
        raise KeyError("Key not found: " + str(key))
    return self.json_data[key]

def __str__(self):
    return str(self.json_data)

def __post_init__(self):
    "
    Post init function that load the json data from the file and set the attributes.
    "
    if self.json_data is None:
        self.json_data = self.load_json_data()
    for key, value in self.json_data.items():
        setattr(self, key, value)
"""
