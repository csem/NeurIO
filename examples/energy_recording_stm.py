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
from neurio.devices.callbacks import PowerProfilerCallback
from neurio.devices.runner import Runner

import tensorflow as tf
import numpy as np
import time
import os

# for MacOS
user = os.environ["USER"]
os.environ["STM32CUBEPROGRAMER_CLI_PATH"] = os.path.expanduser(
    "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI")
os.environ["X_CUBE_AI_PATH"] = os.path.expanduser(f"/Users/{user}/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0")
os.environ["STM32CUBEIDE_PATH"] = os.path.expanduser("/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE")

# define model
input_shape = (1, 28, 28, 1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape[1:]),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.summary()

input_data = np.random.random((4, 28, 28, 1)).astype(np.float32)

log_dir = os.path.join("logs-{}".format(time.strftime("%Y%m%d-%H%M%S")))
stm_device = STM32(port='serial', device_identifier="STM32L4R9", name='stm32', verbose=1, log_dir=log_dir)
#ppk2 = PowerProfilerKitII(port='/dev/cu.usbmodemF89AE991B16A2', name='ppk2', acquisition_interval=0.01, verbose=1, log_dir=log_dir)

runner = Runner(stm_device, callbacks=[
    #PowerProfilerCallback(ppk2)
])

runner.prepare_for_inference(model)
pred, prof = runner.infer(input_x=input_data, batch_size=1)

print(pred)