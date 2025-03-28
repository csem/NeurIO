#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 15.01.2024
Description: TODO
"""
import os
import sys

sys.path.append("../")
import json
from tflite_runtime.interpreter import Interpreter
from neurio.devices.physical.google.coral import EdgeTPU
from neurio.devices.runner import Runner
from neurio.devices.monitors.power_monitor import PowerProfilerKitII
from neurio.devices.callbacks import PowerProfilerCallback
import numpy as np
import torch


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        If input object is an ndarray, it will be converted into a list.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


filepath = "~/Documents/Projects/ANDANTE/facedetection-bench/exp_ANDANTE_D512/FAE/checkpoints/AvgConvNet5_100.0ms_merge_polarity_True_num_classes_1_augment_p_0.5_alpha_1.0_cutout_16_init_lr_0.001_l1_0_l2_0_synops_0_objsyn_0/best_model_edgetpu.tflite"
filepath = os.path.expanduser(filepath)

saved_datasets = os.path.expanduser("~/Documents/Data/cache/numpy_datasets/FAE/1e5_first_frame_only")
# Get input data
inputs = np.load(os.path.join(saved_datasets, "test_inputs.npy")).astype(np.float32)
labels = np.load(os.path.join(saved_datasets, "test_labels.npy")).astype(np.float32)

inputs = np.moveaxis(inputs, 1, -1)

device = EdgeTPU(port='usb', name='edgetpu', verbose=1)
# ppk2 = PowerProfilerKitII(port=None, name='ppk2', acquisition_interval=0.0001, mode="source", source_voltage=5000,
# verbose=1, log_dir=f"edgeTPULog-{os.path.basename(file)}")

runner = Runner(device, callbacks=[
    # PowerProfilerCallback(ppk2)
])
runner.prepare_for_inference(filepath)
pred, profiler = runner.infer(input_x=inputs, batch_size=1)

res = np.asarray(profiler["inference"]["inference_times"][1:]) * 1000

results = {"y_pred": pred,
           "y_true": labels,
           "profiler": profiler.json_data
           }

with open(os.path.dirname(filepath) + "/edgetpu_inference_results.json", "w") as f:
    json.dump(results, f, cls=NumpyArrayEncoder)
