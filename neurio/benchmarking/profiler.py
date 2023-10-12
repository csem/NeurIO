#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 05.10.23
Description: Dataclass for the profiler
"""
import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

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

    def __init__(self, json_file=None):
        if json_file is None:
            self.json_data = dict(base_skeleton_dictionnary)
        else:
            if isinstance(json_file, str):
                self.json_file = json_file
                self.json_data = self.load()
            elif isinstance(json_file, dict):
                self.json_data = json_file
            else:
                raise ValueError("json_file must be a path to a json file or a dict")

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
                    if x is None or len(x)==0:
                        return y
                    elif y is None or len(y)==0:
                        return x
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
                        return [x,y]

        def merge_two_profilers(x, y):
            x = x.__to_dict__()
            y = y.__to_dict__()

            res = merge_two_dicts(x, y)

            new_p = Profiler(res)
            return new_p

        if not isinstance(other, list):
            other = [other]
        profiler = other[0]
        for p in other[1:]:
            profiler = merge_two_profilers(profiler, p)

        return profiler

    def load(self) -> Dict[str, Any]:
        """
        Load json data from file
        :return: Dict[str, Any] containing the json data
        """
        with open(self.json_file, 'r*') as file:
            self.json_data = json.load(file)
        return self.json_data

    def save(self, filepath: str):
        """
        Save json data to file
        :param outfile: where to save the data of the profiler, as json file
        """
        if not filepath.endswith('.json'):
            raise ValueError('The path must be a json file')
        with open(filepath, 'w') as file:
            json.dump(self.json_data, file)

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
    def __post_init__(self):
        "
        Post init function that load the json data from the file and set the attributes.
        "
        if self.json_data is None:
            self.json_data = self.load_json_data()
        for key, value in self.json_data.items():
            setattr(self, key, value)
    """
