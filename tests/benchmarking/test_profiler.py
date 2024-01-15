#!/user/bin/env python

from neurio.benchmarking.profiler import Profiler
import tensorflow as tf


def test_merge():
    k210_profiler = Profiler()

    k210_profiler["model"] = {
        "model_name": "TEST_MODEL",
        "framework": "tensorflow",
        "model_datetime": "Tue Oct 05 16:00:00 2021",
        "compile_datetime": "Tue Oct 05 16:00:00 2021",
        "inputs": [{"name": "input_1", "shape": [1, 28, 28, 1]}
                   ],
        "outputs": [{"name": "output_1", "shape": [1, 10]}
                    ],
        "parameters": 1928491,
        "macc": 245129,
    }

    k210_profiler["device"] = {
        "name": "TEST_DEVICE",
        "port": "TEST_PORT",
        "dev_type": "TEST_DEV_TYPE",
        "desc": "TEST_DESC",
        "dev_id": "TEST_DEV_ID",
        "system": "TEST_SYSTEM",
        "sys_clock": 400000000,
        "bus_clock": 400000000,
        "attrs": ["TEST_ATTR_1", "TEST_ATTR_2",
                  ]
    }

    k210_profiler["runtime"] = {
        "name": "TEST_RUNTIME",
        "version": [0, 1],
        "tools_version": {"tensorflow": [int(x) if x.isnumeric() else x for x in tf.__version__.split(".")],
                          "maixpy": [0, 6, 2],
                          # TODO automatically get from board"ncc": [0, 2, 0, "beta4"],"kmodel": "v4",
                          }
    }

    k210_profiler["inference"] = {
        "batch_size": None,
        "model_load_time": None,
        "preprocess_time": [],
        "inference_time": [],
    }

    profiler2 = Profiler()
    profiler2["inference"] = {
        "batch_size": 4,
        "model_load_time": 0.1,
        "preprocess_time": [0.1, 0.2, 0.3, 0.4],
        "inference_time": [0.1, 0.2, 0.3, 0.4],
    }

    new_profiler = Profiler.merge([k210_profiler, profiler2])

    # check new profiler has all the keys
    for key in list(k210_profiler.__to_dict__().keys()):
        assert key in list(new_profiler.__to_dict__().keys())

    # check that fixed values stay unchanged (and are not transformed to lists)
    assert new_profiler["model"] == k210_profiler["model"]
    assert new_profiler["device"] == k210_profiler["device"]
    assert new_profiler["runtime"] == k210_profiler["runtime"]

    assert new_profiler["inference"]["batch_size"] == 4
    assert new_profiler["inference"]["model_load_time"] == 0.1
    assert new_profiler["inference"]["preprocess_time"] == [0.1, 0.2, 0.3, 0.4]
    assert new_profiler["inference"]["inference_time"] == [0.1, 0.2, 0.3, 0.4]

    # merge with a third profiler

    profiler3 = Profiler()
    profiler3["inference"] = {
        "batch_size": 4,
        "model_load_time": 0.1,
        "preprocess_time": [0.3, 0.3, 0.3, 0.3],
        "inference_time": [0.5, 0.5, 0.5, 0.5],
    }

    new_profiler = Profiler.merge([profiler2, profiler3])
    assert new_profiler["inference"]["preprocess_time"] == [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.3, 0.3]
    assert new_profiler["inference"]["inference_time"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5]
    assert new_profiler["inference"]["batch_size"] == 4
    assert new_profiler["inference"]["model_load_time"] == 0.1


def test_merge_profiler_batch_1_with_dict():
    A_dict = {
        "profiler_version": "1.0.0",
        "model": {"model_name": None, "framework": None, "model_datetime": None, "compile_datetime": None,
                  "inputs": None, "outputs": None, "parameters": None, "macc": None, },
        "device": {"name": None, "port": None, "dev_type": None, "desc": None, "dev_id": None, "system": None,
                   "sys_clock": None, "bus_clock": None, "attrs": []},
        "runtime": {"name": None, "version": [], "tools_version": {}},

        "inference": {"batch_size": None, "model_load_time": None, "preprocess_time": [1], "inference_time": [2]},
    }

    B_dict = {
        "profiler_version": "1.0.0",
        "model": {"model_name": None, "framework": None, "model_datetime": None, "compile_datetime": None,
                  "inputs": None, "outputs": None, "parameters": None, "macc": None, },
        "device": {"name": None, "port": None, "dev_type": None, "desc": None, "dev_id": None, "system": None,
                   "sys_clock": None, "bus_clock": None, "attrs": []},
        "runtime": {"name": None, "version": [], "tools_version": {}},

        "inference": {"batch_size": None, "model_load_time": None, "preprocess_time": [2], "inference_time": [3]},
    }

    prof1 = Profiler(A_dict)
    prof2 = Profiler(B_dict)
    prof3 = Profiler.merge([prof1, prof2])
    assert prof3["inference"]["preprocess_time"] == [1, 2]
    assert prof3["inference"]["inference_time"] == [2, 3]

def test_merge_profiler_batch_1_with_set_attr():

    prof1 = Profiler()
    prof1["inference"]["preprocess_time"] = [1]
    prof1["inference"]["inference_time"] = [2]
    prof2 = Profiler()
    prof2["inference"]["preprocess_time"] = [2]
    prof2["inference"]["inference_time"] = [3]
    prof3 = Profiler.merge([prof1, prof2])
    assert prof3["inference"]["preprocess_time"] == [1, 2]
    assert prof3["inference"]["inference_time"] == [2, 3]