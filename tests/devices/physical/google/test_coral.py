#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 07.01.2024
Description: TODO
"""

import numpy as np
from devices.physical.google.coral import EdgeTPU
import os
from tensorflow.keras import layers, models

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

def test_edgeTPU_pipeline():
    input_data = np.random.random((10, 32, 32, 1))
    model = lenet_5()

    device = EdgeTPU(port=None, log_dir=None)
    device.prepare_for_inference(model=model)
    predictions = device.predict(input_x=input_data, batch_size=2)
    print(predictions)