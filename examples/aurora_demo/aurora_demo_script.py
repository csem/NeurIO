#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 13.01.2024
Description: TODO
"""

from neurio.devices.physical.st.stm32 import STM32
from neurio.devices.monitors.power_monitor import PowerProfilerKitII
from neurio.devices.callbacks import PowerProfilerCallback, TimerSyncCallback
from neurio.devices.runner import Runner

import tensorflow as tf
import numpy as np
import time
import sys
import os

# for MacOS
user = os.environ["USER"]
os.environ["STM32CUBEPROGRAMER_CLI_PATH"] = os.path.expanduser(
    "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI")
os.environ["X_CUBE_AI_PATH"] = os.path.expanduser(f"/Users/{user}/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0")
os.environ["STM32CUBEIDE_PATH"] = os.path.expanduser("/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE")

import pickle

# load pickle
with open("./assets/eval_dataset.pkl", "rb") as f:
    data = pickle.load(f)

input_data = data["audio_data_tensor"]
input_data = np.expand_dims(input_data, axis=-1)
labels_data = data["labels_tensor"]



# define model
model_file = "./assets/rain_binary_filter_s10_2d.h5"
model = tf.keras.models.load_model(model_file)

log_dir = os.path.join("logs-STM32L4R9")
stm_device = STM32(port='serial', device_identifier="STM32L4R9", name='stm32', verbose=1, log_dir=log_dir)
#ppk2 = PowerProfilerKitII(port='/dev/cu.usbmodemF89AE991B16A2', name='ppk2', acquisition_interval=0.01, verbose=1, log_dir=log_dir)

runner = Runner(stm_device, callbacks=[
    #TimerSyncCallback(log_dir=log_dir, name="time_sync")
    #PowerProfilerCallback(ppk2)
])

runner.prepare_for_inference(model_file, quantize=True)


pred, prof = runner.infer(input_x=input_data, batch_size=1)

print(pred)