#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 21.09.22
Description: Convert Keras model to TFLite
"""
import os.path

import keras
import tensorflow as tf
from typing import Union


def keras_to_tflite(keras_model: Union[keras.Model, str], tflite_path: str, custom_objects: dict = {}) -> str:
    if isinstance(keras_model, str) and keras_model.endswith(".tflite"):
        print("Already converted to tflite.")
        tflite_path = keras_model
    else:
        if isinstance(keras_model, str) and keras_model.endswith(".h5"):
            keras_model = tf.keras.models.load_model(keras_model, custom_objects=custom_objects)
        elif isinstance(keras_model, keras.Model):
            keras_model = keras_model
        else:
            raise Exception("Model should be a path to a .h5 file or a tf.keras model.")

        save_folder = os.path.join(tflite_path.replace(".tflite", ""))
        keras_model.save(save_folder)
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(save_folder)  # path to the SavedModel directory
        tflite_model = converter.convert()

        # Save the model.
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

    return tflite_path


def tf_to_tflite(saved_model_dir: str, tflite_path: str) -> str:
    if not isinstance(saved_model_dir, str):
        raise Exception("Model should point to directory of saved model.")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
    tflite_model = converter.convert()

    # Save the model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_path
