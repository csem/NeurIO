#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 15.01.2024
Description: TODO
"""
import os
import json
from tflite_runtime.interpreter import Interpreter
from neurio.devices.physical.google.coral import EdgeTPU
from neurio.devices.runner import Runner
from neurio.devices.monitors.power_monitor import PowerProfilerKitII
from neurio.devices.callbacks import PowerProfilerCallback
import numpy as np

root = os.path.expanduser("./assets")
storage_folder = os.path.join(root, "results")
os.makedirs(storage_folder, exist_ok=True)
files = [os.path.join(root, x) for x in os.listdir(root) if x.endswith(".tflite")]
print(files)

input_data = np.random.random((101, 224, 224, 3)).astype(np.float32)

results = {}

for file in files:
    device = EdgeTPU(port='usb', name='edgetpu', verbose=1)
    ppk2 = PowerProfilerKitII(port=None, name='ppk2', acquisition_interval=0.0001, mode="source", source_voltage=5000,
                              verbose=1, log_dir=f"edgeTPULog-{os.path.basename(file)}")

    runner = Runner(device, callbacks=[
        PowerProfilerCallback(ppk2)
    ])
    runner.prepare_for_inference(file)
    pred, profiler = runner.infer(input_x=input_data, batch_size=1)
    res = np.asarray(profiler["inference"]["inference_times"][1:])*1000

    print(f"Mean inference time: {np.mean(res)} ms")
    if file not in results:
        results[file] = []

    results[file].append(np.mean(res))
    print("DONE")