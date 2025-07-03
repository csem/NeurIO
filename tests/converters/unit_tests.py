#!/usr/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2025
Creation: 03.07.2025
Description: Unit tests for the ModelConverter class
"""
import keras
import pytest
from unittest.mock import patch, MagicMock
from neurio.converters.model_converter import ModelConverter

# Fixtures and mock setup
@pytest.fixture
def mock_keras_model():
    return MagicMock()

@pytest.fixture
def converter():
    return ModelConverter()

def test_unsupported_format_raises(converter):
    with pytest.raises(ValueError, match="Unsupported output format: onnx"):
        converter.convert(model="model.h5", output_format="onnx")

def test_invalid_output_extension_raises(converter):
    with pytest.raises(ValueError, match="Output path must end with .tflite"):
        converter.convert(model="model.h5", output_path="model.invalid")

def test_missing_output_path_for_keras_model_raises(converter, mock_keras_model):
    with pytest.raises(ValueError, match="Output path must be specified"):
        converter.convert(model=mock_keras_model)

@patch("neurio.converters.tflite_utils.keras_to_tflite")
def test_convert_keras_model_to_tflite(mock_convert, converter):
    keras_model = keras.Sequential()
    mock_convert.return_value = "model.tflite"
    result = converter.convert(model=keras_model, output_path="model.tflite")
    mock_convert.assert_called_once()
    assert result == "model.tflite"

@patch("neurio.converters.tflite_utils.onnx_to_tflite")
def test_convert_onnx_model_to_tflite(mock_convert, converter):
    mock_convert.return_value = "model.tflite"
    result = converter.convert(model="model.onnx", output_path="model.tflite")
    mock_convert.assert_called_once_with("model.onnx", "model.tflite")
    assert result == "model.tflite"

@patch("neurio.converters.tflite_utils.keras_to_tflite")
def test_convert_keras_h5_file_to_tflite(mock_convert, converter):
    mock_convert.return_value = "model.tflite"
    result = converter.convert(model="model.h5", output_path="model.tflite")
    mock_convert.assert_called_once_with("model.h5", "model.tflite")
    assert result == "model.tflite"

def test_unsupported_input_format_raises(converter):
    with pytest.raises(ValueError, match="Unsupported model file format"):
        converter.convert(model="model.pb", output_path="model.tflite")
