#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 05.10.23
Description: Block based benchmarks
"""
from abc import ABC, abstractmethod
from neurio.devices.device import Device
import json
import numpy as np
import tensorflow as tf
#from tensorflow.keras import Model, Sequential
#from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, DepthwiseConv2D, Dense, AveragePooling1D, AveragePooling2D, \
 #   MaxPooling1D, MaxPooling2D, Activation, BatchNormalization, Reshape, ZeroPadding1D, ZeroPadding2D, Activation


class Benchmark:
    """
    Base class for benchmarks
    """

    def __int__(self):
        pass

    @abstractmethod
    def run_on(self, device: Device):
        """
        Run the Benchmark on a target device.

        :param device: the target device
        :return: the results as JSON
        """
        raise NotImplementedError


class LayerBasedBenchmark(Benchmark):
    """
    Benchmarking suite to test the inference of single layers on a given device.

    The list of layer is as follows:

    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Block            | Config-1              | Config-2              | Config-3              | Config-4              |
    +==================+=======================+=======================+=======================+=======================+
    | Convolutional 1D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Convolutional 2D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Convolutional 2D | filters 32, 1x1       | filters 32, 3x3       | filters 32, 5x5      | filters 64, 7x7        |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Depthwise 2D conv| filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Convolutional 3D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Fully connected  | 128-32 neurons        | 256-64 neurons        | 512-128 neurons       | 1024-256 neurons      |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | AvgPooling1D     | 2, stride 1           | 2, stride 2           | 4, stride 2           | 7, stride 1           |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | AvgPooling2D     | 2x2, stride 1         | 2x2, stride 2         | 4x4, stride 2         | 7x7, stride 1         |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | MaxPooling1D     | 2, stride 1           | 2, stride 2           | 4, stride 2           | 7, stride 1           |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | MaxPooling2D     | 2x2, stride 1         | 2x2, stride 2         | 4x4, stride 2         | 7x7, stride 1         |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Activation       | ReLU                  | Sigmoid               | ReLU6                 | Tanh                  |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Batch norm       | 1D                    | 2D                    | 3D                    | 4D                    |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    | Reshape          | 1D                    | 2D                    | 3D                    | 4D                    |
    +------------------+-----------------------+-----------------------+-----------------------+-----------------------+
    """

    def __init__(self, seed=0, num_inference=16):
        """
        Constructor of the LayerBasedBenchmark.

        :param seed: the seed to intialize the weights of the models and the mock data.
        :param num_inference: number of (random) inference per model.
        """
        super().__init__()

        configuration_path = "assets/single_layers.json"
        with open(configuration_path) as f:
            self.configurations = json.load(f)

        self.num_inference = num_inference
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed + 1)

    def __instanciate_layer__(self, layer):
        """
        Instantiate the layer given its configuration.

        :param layer: a json describing the layer type and its parameters
        :return: an instance of tf.keras.layers.Layer
        """
        params = layer["Parameters"]
        if layer["Layer"] == "Conv1D":
            return Conv1D(filters=params["filters"], kernel_size=params["kernel_size"], strides=params["strides"],
                          padding=params["padding"])
        elif layer["Layer"] == "Conv2D":
            return Conv2D(filters=params["filters"], kernel_size=params["kernel_size"], strides=params["strides"],
                          padding=params["padding"])
        elif layer["Layer"] == "Conv3D":
            return Conv3D(filters=params["filters"], kernel_size=params["kernel_size"], strides=params["strides"],
                          padding=params["padding"])
        elif layer["Layer"] == "DepthwiseConv2D":
            return DepthwiseConv2D(depth_multiplier=params["filters"], kernel_size=params["kernel_size"],
                                   strides=params["strides"],
                                   padding=params["padding"])
        elif layer["Layer"] == "Dense":
            return Dense(units=params["units"])
        elif layer["Layer"] == "AveragePooling1D":
            return AveragePooling1D(pool_size=params["pool_size"], strides=params["strides"], padding=params["padding"])
        elif layer["Layer"] == "AveragePooling2D":
            return AveragePooling2D(pool_size=params["pool_size"], strides=params["strides"], padding=params["padding"])
        elif layer["Layer"] == "MaxPooling1D":
            return MaxPooling1D(pool_size=params["pool_size"], strides=params["strides"], padding=params["padding"])
        elif layer["Layer"] == "MaxPooling2D":
            return MaxPooling2D(pool_size=params["pool_size"], strides=params["strides"], padding=params["padding"])
        elif layer["Layer"] == "Activation":
            return Activation(activation=params["activation"])
        elif layer["Layer"] == "BatchNormalization":
            return BatchNormalization()
        elif layer["Layer"] == "Reshape":
            return Reshape(target_shape=params["target_shape"])
        else:
            raise ValueError("Unknown layer: " + layer["Layer"])

    def __create_model__(self, config):
        """
        Create a Keras model with a single layer

        :param line: line index of the configuration
        :param config: configuration index of the configuration
        :return: Keras model containing a single layer.
        """

        layers = config["Layers"]
        model_layers = []
        for l in layers:
            instanciated_layer = self.__instanciate_layer__(l)
            model_layers.append(instanciated_layer)

        model = Sequential(model_layers)
        shape = [None, *config["Input_shape"]]  # In case of dense block, use the first connected layer as input
        model.build(input_shape=shape)

        model.compile()
        return model

    def get_model(self, x, y):
        """
        Get the model at the location x,y of the LayerBasedBenchmark

        :param line_idx:
        :param config_idx:
        :return:
        """
        conf = self.configurations["configurations"][x][y]
        model = self.__create_model__(conf)
        model.summary()

    def info(self, x=None, y=None):
        """
        Prints the structure of the LayerBasedBenchmark.

        :param line_idx: If set, prints all models in the line x from the LayerBasedBenchmark
        :param config_idx: If set, prints the model located at the line x and column y provided.
        """

        line_idx = x
        config_idx = y
        if line_idx is None and config_idx is not None:
            raise ValueError("If config is specified, line must be specified too")
        elif line_idx is not None and config_idx is None:
            conf = self.configurations["configurations"][line_idx]
        elif line_idx is not None and config_idx is not None:
            conf = self.configurations["configurations"][line_idx][config_idx]
        else:
            conf = self.configurations["description"]

        if isinstance(conf, str):
            print(conf)  # print description
        else:
            # pretty print json configuration
            print(json.dumps(conf, indent=4, sort_keys=True))

    def run_on(self, device: Device):
        """
        Run the Benchmark on a target device.
        Each model of the benchmark is compiled and deployed on the device for inference.
        For the inference, random data is generated according to the dimension of the input, and infered using the device `infer` method.

        :param device: a device inheriting from `Device`
        :return: the results as JSON
        """
        all_models = self.configurations["configurations"]

        for i in range(len(all_models)):
            configs_layer = all_models[i]
            for j in range(len(configs_layer)):
                tf.keras.backend.clear_session()
                configuration_model = configs_layer[j]
                model = self.__create_model__(configuration_model)

                device.prepare_for_inference(model)
                data = np.random.random([self.num_inference, *configuration_model["Input_shape"]])
                y_pred, profile = device.predict(data)

                configuration_model["results"] = profile
        return all_models


class MockDevice(Device):

    def __init__(self):
        self.name = "dummy"

    def prepare_for_inference(self, model: tf.keras.Model):
        self.model = model

    def predict(self, data: np.array, batch_size: int = 32, return_stats: bool = False, verbose: bool = True):
        predictions = []
        times = []
        for i in range(len(data)):
            s = time.time()
            y_pred = self.model.predict(data[i:i + 1])
            predictions.append(y_pred)
            times.append(time.time() - s)

        profiler = {
            "Inference time": times
        }

        return y_pred, profiler


if __name__ == '__main__':
    benchmark = LayerBasedBenchmarking()
    device = MockDevice()

    results = benchmark.run_on(device)
    print(results)
