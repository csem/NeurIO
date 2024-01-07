#!/user/bin/env python

import numpy as np
from tensorflow.keras import layers, models
from devices.physical.st.stm32 import STM32
import os

# for MacOS
user = os.environ["USER"]
os.environ["STM32CUBEPROGRAMER_CLI_PATH"] = os.path.expanduser(
    "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI")
os.environ["X_CUBE_AI_PATH"] = os.path.expanduser(f"/Users/{user}/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0")
os.environ["STM32CUBEIDE_PATH"] = os.path.expanduser("/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE")

def lenet_5():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

def test_stm_pipeline():
    device = "NUCLEO-H723ZG"
    input_data = np.random.random((10, 32, 32, 1))
    model = lenet_5()

    device = STM32(port=None, device_identifier=device, log_dir=None)
    device.prepare_for_inference(model=model)
    predictions = device.predict(input_x=input_data, batch_size=2)
    print(predictions)