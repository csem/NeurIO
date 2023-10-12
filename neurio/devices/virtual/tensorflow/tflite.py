#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 06.02.23
Description: Evaluator for the TFLite models
"""

# Imports
import json
import multiprocessing
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from neurio.helpers import NpEncoder
from neurio.devices.device import Device
from neurio.common import Prediction
from neurio.converters.tflite_utils import keras_to_tflite
import os
import platform

import psutil


# supported model type
supported_types = [
    'tflite',
    "h5"
]


class TFLiteVirtual(Device):
    # Benchmark function
    def __init__(self, port: any, name: str = "", log_dir: str = None, options: dict = {'is_quantized': True}):
        super().__init__(port, name, log_dir)
        self.is_quantized = options['is_quantized']
        self.model = None
        self.is_ready_for_inference = False
        raise Exception("TFLiteVirtual is not supported - old implemetation")

    def is_alive(self):
        if self.interpreter is None:
            return False
        else:
            return True

    def create_log_dirs(self):
        pass

    def prepare_for_inference(self, model: tf.keras.Model, options: dict = {}):
        self.model_datetime = time.strftime("%a %b %d %H:%M:%S %Y")
        self.model = model

        if isinstance(model, tf.keras.Model):
            self.model_name = os.path.join(self.log_dir, model.name)
            model.save(self.model_name, save_format="h5")
        elif isinstance(model, str):
            self.model_name = model

        if not self.is_ready_for_inference:
            self.tflite_model = keras_to_tflite(model, self.model_name.replace(".h5", ".tflite"))
            self.compile_datetime = time.strftime("%a %b %d %H:%M:%S %Y")


            self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model)
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']

            self.is_ready_for_inference = True
        else:
            print("Already ready for inference")

    def save_data(self, data: any, location=None):
        raise NotImplementedError()

    def detach(self):
        if self.is_ready_for_inference:
            self.tflite_model = None
            self.interpreter = None
            self.input_details = None
            self.output_details = None
            self.input_shape = None
            self.is_ready_for_inference = False
        else:
            print("Already detached")

    def predict(self, data: any, batch_size: int = 32, verbose: bool = True) -> Prediction:
        """
        Predicts the data
        :param data: input data to predict
        :param batch_size: batch size for the prediction
        :param return_stats: return statistics of the prediction
        :param verbose: whether to print the progress bar
        :return:
        """
        input_dataset = data

        try:
            # load the dataset
            if input_dataset is None:
                # Test the model on random input data.
                input_data = np.array(np.random.random_sample(self.input_shape), dtype=np.float32)
                inputs = np.repeat(input_data, 100, axis=0)
            else:
                if self.is_quantized:
                    inputs = np.load(input_dataset).astype(np.uint8)
                else:
                    inputs = np.load(input_dataset).astype(np.float32)

            # loop over input data
            if batch_size!=1:
                print("Only batch size of 1 is supported. Switching to batch size of 1.")
                batch_size = 1
            nb = batch_size
            all_outputs = []
            times = []
            for i in tqdm(range(0, len(inputs), nb)):
                input_data = inputs[i:i + nb]
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                # Perform the inference
                start = time.time()
                self.interpreter.invoke()
                end = time.time()
                times.append(end - start)
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                all_outputs.append(output_data)

            y_pred = np.asarray(all_outputs)
            #########################################################################################
            #
            #   Post process results
            #
            #########################################################################################

            # Retrive metrics
            c_durations = np.array(times)

            mean_latency = c_durations.mean()
            std_latency = c_durations.std()

            profiler = {}
            profiler['latency'] = mean_latency
            profiler['std_latency'] = std_latency

            # Add other metrics
            profiler['model_name'] = self.model_name.split('.')[0]
            profiler['model_type'] = self.model_name.split('.')[1]
            profiler['device_name'] = self.name
            profiler['whole_dataset'] = True
            profiler['batch_size'] = nb


            # add details about input, clean and add to profiler
            [self.input_details[i].pop("dtype") for i in range(len(self.input_details))]
            [self.output_details[i].pop("dtype") for i in range(len(self.output_details))]
            profiler['inputs'] = [self.input_details[i] for i in range(len(self.input_details))]
            profiler['outputs'] = [self.output_details[i] for i in range(len(self.output_details))]

            profiler['model_datetime'] = self.model_datetime
            profiler['compile_datetime'] = self.compile_datetime
            # profiler['weights'] = TODO self.model.count_params()
            # profiler['activations'] = # TODO
            # profiler['macc'] = # TODO
            profiler['size'] = os.path.getsize(self.model_name)

            profiler["runtime"] = {
                "name": "CSEM TFlite pipeline",
                "version": "0.1",
                "tools_version": {
                    "tensorflow": tf.__version__,
                    "keras": tf.keras.__version__,
                    "numpy": np.__version__
                }
            }

            uname = platform.uname()

            profiler["device"] = {
                "dev_type": uname.system,
                "plaftorm": platform.platform(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "sys_clock": int(psutil.cpu_freq().max * 1e6),
                "attr": {
                    "cpu_count": multiprocessing.cpu_count(),
                    "memory": psutil.virtual_memory().total,
                    "disk": psutil.disk_usage('/').total
                }
            }


            # Save results in output file
            result_file = os.path.join(self.results_dir, "results.json")
            with open(result_file, 'w') as f:
                json.dump(profiler, f, cls=NpEncoder)

            print("\n\n Test completed ! Test report available at : " + str(result_file))

            return Prediction(y_pred, profiler)

            # If an exception occured, remove DOcker container and raise the exception again.

        except Exception as e:
            print(e)
            raise e
