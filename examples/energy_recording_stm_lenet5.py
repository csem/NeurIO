#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 13.01.2024
Description: TODO
"""

from neurio.devices.physical.st.stm32 import STM32
from neurio.devices.callbacks import TimerSyncCallback, PowerProfilerCallback
from neurio.devices.runner import Runner
from neurio.devices.monitors.power_monitor import PowerProfilerKitII

import tensorflow as tf
import numpy as np
import os

# for MacOS
user = os.environ["USER"]
os.environ["STM32CUBEPROGRAMER_CLI_PATH"] = os.path.expanduser(
    "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI")
os.environ["X_CUBE_AI_PATH"] = os.path.expanduser(f"/Users/{user}/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0")
os.environ["STM32CUBEIDE_PATH"] = os.path.expanduser("/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE")

# load model from ./assets/experiments
model = tf.keras.models.load_model("./assets/experiments/lenet_model.h5")

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_test = 30
x_test = x_test[:num_test]
y_test = y_test[:num_test]

input_data = x_test.astype(np.float32)
input_data = input_data.reshape((-1, 28, 28, 1))

log_dir = os.path.join("logs-STM32L4R9-float32-test")
stm_device = STM32(port='serial', device_identifier="STM32L4R9", name='stm32', verbose=1, log_dir=log_dir)
ppk2 = PowerProfilerKitII(port='/dev/cu.usbmodemF89AE991B16A2', name='ppk2', acquisition_interval=0.01, verbose=1, log_dir=log_dir)


runner = Runner(stm_device, callbacks=[
    TimerSyncCallback(log_dir=log_dir, name="time_sync"),
    PowerProfilerCallback(ppk2)

])

runner.prepare_for_inference(model)
pred, prof = runner.infer(input_x=input_data, batch_size=1)

print(pred)