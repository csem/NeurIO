#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 07.01.2024
Description: Google Coral TPU device
"""

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from neurio.devices.device import Device
import os
import warnings
import numpy as np
from neurio.benchmarking.profiler import Profiler
import platform
import tensorflow as tf
import tflite_runtime as tflite
import sys
import time

class EdgeTPU(Device):
    def __init__(self, port: any = "serial", device_identifier: str = "edgetpu", name: str = "edgetpu",
                 log_dir: str = None, **kwargs):

        super().__init__(port, name, log_dir, **kwargs)
        self.verbose = kwargs.get("verbose", 0)
        self.device_identifier = device_identifier

    def __check_env__(self):
        os.system()

    def __prepare_model__(self, model, quantize=False, calibration_inputs=None):
        # save in logdir
        self.model_path = os.path.join(self.log_dir, model.name)
        self.tflite_path = os.path.join(self.log_dir, f"{model.name}.tflite")

        # save model
        model.save(self.model_path)

        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantize and calibration_inputs is None:
            raise ValueError("Calibration generator must be provided for quantization")
        if calibration_inputs is not None:
            converter.representative_dataset = calibration_inputs
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        # Save the TFLite model
        with open(self.tflite_path, "wb") as f:
            f.write(tflite_model)

    def __generate_inference_code__(self):
        warnings.warn("Generating inference code is not necessary for EdgeTPU devices")
        pass

    def __deploy_model__(self):
        sys.path.append("/usr/local/lib/")
        delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
        #interpreter = make_interpreter(self.tflite_path)
        import tflite_runtime.interpreter as tflite_runtime
        delegate = tflite_runtime.Delegate(os.path.join("/usr","local", "lib", "libedgetpu.1.dylib"))
        interpreter = tflite_runtime.Interpreter(
            model_path=self.tflite_path,
            experimental_delegates=[delegate])

        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.model_input_size = common.input_size(self.interpreter)
        self.model_quantize_params = common.input_details(self.interpreter, 'quantization_parameters')
        self.model_quantize_scale = self.model_quantize_params['scales']
        self.model_quantize_zero_point = self.model_quantize_params['zero_points']

    def __deploy_code__(self):
        pass

    def __prepare_data__(self, input_x, **kwargs):
        new_x = []
        self.model_quantize_mean = np.mean(input_x)
        self.model_quantize_std = np.std(input_x)

        for image in input_x:
            if self.model_quantize_params['quantized_dimension'] != 0:
                if abs(self.model_quantize_scale * self.model_quantize_std - 1) < 1e-5 and abs(
                        self.model_quantize_mean - self.model_quantize_zero_point) < 1e-5:
                    # Input data does not require preprocessing.
                    new_x.append(image)
                else:
                    # Input data requires preprocessing
                    normalized_image = np.asarray(image).astype(np.float32) - self.model_quantize_mean
                    normalized_image /= (self.model_quantize_std * self.model_quantize_scale)
                    normalized_image += self.model_quantize_zero_point

                    np.clip(normalized_image, 0, 255, out=normalized_image)
                    new_x.append(normalized_image.astype(np.uint8))
            else:
                new_x.append(image)
        return new_x

    def __transfer_data_to_memory__(self, input_x):
        if len(input_x) != 1:
            raise ValueError("EdgeTPU devices only support batch size 1")
        # copy data to memory
        start = time.perf_counter()
        common.set_input(self.interpreter, input_x[0])
        self.transfer_time = time.perf_counter() - start

    def __run_inference__(self, profile: bool = True):
        start = time.perf_counter()
        self.interpreter.invoke()
        self.inference_time = time.perf_counter() - start

    def __read_inference_results__(self):
        # readout
        predictions = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])  # get tensor

        # create lists for
        inference = self.inference_time
        load = self.transfer_time
        preprocess = 0  # no preprocessing happening on board

        # store in profile as lists of 1 batch
        temp_profiler = Profiler()
        temp_profiler.inference_times = [inference]
        temp_profiler.load_times = [load]
        temp_profiler.preprocess_times = [preprocess]

        return predictions, temp_profiler

    def is_alive(self, timeout: int = 20) -> bool:
        pass

    def __str__(self):
        pass
