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
from neurio.converters import tflite_utils
from typing import Union

SUPPORTED_FORMATS = ['tflite']  # TODO add onnx, keras, tensorflow, pytorch, etc.

class ModelConverter:
    """A universal converter that accepts different model formats and converts them to the desired output format"""

    def __init__(self):
        pass

    @staticmethod
    def convert(model=Union[keras.Model, str], output_format: str = "tflite", output_path: str = None, **kwargs) -> str:
        """
        Convert a model to the desired format.

        :param model: The model to convert. It can be a Keras model or a path to a model file (onnx, tflite, tensorflow, keras).
        :param output_format: The format to convert the model to (e.g., 'tflite', 'onnx', 'keras', 'tensorflow', 'pytorch').
        :param output_path: The path where the converted model will be saved.
        :param kwargs: Additional parameters for the conversion.
        :return: The path to the converted model.
        """
        if output_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}. Currently supported formats are: {SUPPORTED_FORMATS}")

        if not isinstance(model, str) and output_path is None:
            raise ValueError("Output path must be specified when model is not a string path.")

        if output_path is not None and not output_path.endswith(f".{output_format}"):
            raise ValueError(f"Output path must end with .{output_format} extension. Provided: {output_path}")

        # strategy: convert everything to tflite, then to desired format
        tflite_path = None
        if isinstance(model, keras.Model):
            if output_path is None:
               raise ValueError("Output path must be specified when model is a Keras model.")

            tflite_path = tflite_utils.keras_to_tflite(model, output_path)

        elif isinstance(model, str):
            if model.endswith(".tflite"):
                tflite_path = model
            elif model.endswith(".onnx"):
                tflite_path = tflite_utils.onnx_to_tflite(model, output_path)
            elif model.endswith(".h5"):
                tflite_path = tflite_utils.keras_to_tflite(model, output_path)
            else:
                raise ValueError(f"Unsupported model file format: {model}")
        return tflite_path



