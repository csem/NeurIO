#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 12.10.22
Description: Utilitary functions
"""

import os
import sys

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)
import json
import numpy as np

with open(os.path.join(dir_name, "resources/common/profiler_skeleton.json"), "r") as f:
    profiler_skeleton = json.load(f)


def get_null_profiler(profiler: any):
    if not isinstance(profiler, dict):
        raise ValueError("Profiler object should a dictionnary")

    null_fields = []
    for k in list(profiler.keys()):
        d = profiler[k]
        if isinstance(d, dict):
            if len(d) == 0:
                null_fields.append(k)
            else:
                null_fields_d = get_null_profiler(d)
                null_fields += null_fields_d

        elif isinstance(d, list):
            if len(d) == 0:
                null_fields.append(k)
        elif d is None:
            null_fields.append(k)

    return null_fields


def check_profiler(profiler: any):
    """
    Check if profiler is valid, by looking at the profiler fields.
    :param profiler: profiler object
    :raise ValueError: if profiler is not valid
    """

    null_fields = get_null_profiler(profiler)
    if len(null_fields) != 0:
        raise ValueError("Please provide values for the following fields: {}".format(null_fields))

    # check all fields are valid
    for k in profiler_skeleton:
        if k not in profiler:
            raise ValueError("Value '{}' is absent from the profiler".format(k))


class NpEncoder(json.JSONEncoder):
    """
    Numpy encoder for json files
    """

    def default(self, obj):
        """
        Default method for encoding
        :param obj: object to encode (can be numpy)
        :return: encoded object as json
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def print_iterator(it: iter) -> None:
    """
    Print incoming data from iterator utile the iterator is empty.
    :param it: iterator to print
    """
    old = ''
    while True:
        try:
            new = next(it).decode("utf-8")
            if old != new:
                print(new)
                old = new

                # Check for errors
                if "Build Failed" in new:
                    raise RuntimeError('Failed to compile the project')
                if "Error: File does not exist" in new:
                    raise RuntimeError('.elf file does no exist')
                if "No debug probe detected" in new:
                    raise RuntimeError('Device not detected')
                if "overflowed" in new:
                    raise RuntimeError('Model exceeds memory limits')
                if "Operation exceeds memory limits" in new:
                    raise RuntimeError('Model exceeds memory limits')
                if "failed to download the File" in new:
                    raise RuntimeError('failed to download the File (Maybe the device was disconnected during upload)')
        except StopIteration:
            break
