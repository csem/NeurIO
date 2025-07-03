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

import tf2onnx
from onnx_tf.backend import prepare
import onnx

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


def onnx_to_tflite(onnx_file: str, tflite_path: str = None) -> str:
        """Convert ONNX model to TFLite, saving an intermediate SavedModel instance using Tensorflow."""

        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"ONNX file {onnx_file} does not exist.")
        if tflite_path is None:
            tflite_path = onnx_file.replace(".onnx", ".tflite")

        # tensorflow intermediate path
        saved_model_dir = onnx_file.replace(".onnx", "_saved_model")


        onnx_model = onnx.load(onnx_file)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")
        return tflite_path