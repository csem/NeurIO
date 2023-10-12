#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 21.09.22
Description: Converter for models
"""
import os

import keras
import tensorflow as tf
import kendryte_utils
import tflite_utils


class Converter:
    def __init__(self, model: keras.Model, format: str = "keras"):
        self.model = model
        self.format = format

    def to_kmodel(self, save_directory: str, target: str = "k210") -> str:
        os.makedirs(save_directory, exist_ok=True)
        # download required plaforms
        tflite_path = tflite_utils.keras_to_tflite(self.model,
                                                   os.path.join(save_directory, self.model.name + ".tflite"))
        kmodel_path = kendryte_utils.convert_to_kmodel(tflite_path=tflite_path, target=target)
        return kmodel_path


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")])
model.build(input_shape=(None, 28, 28, 3))

conv = Converter(model)
path = conv.to_kmodel("./here")
print(path)

with open(path,"rb") as f:
    print(f.read())