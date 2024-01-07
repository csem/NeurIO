#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 21.09.22
Description: Interface of device
"""
import os
from abc import ABC, abstractmethod
import json

import keras
import numpy as np
import sys
import tensorflow as tf
from neurio.benchmarking.profiler import Profiler
from neurio.exceptions import DeviceNotReadyException
from datetime import datetime

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)


class Device():
    """
    Superclass for all devices.
    """

    def __init__(self, port: any, name: str = "", log_dir: str = None, **kwargs):
        """
        Constructor of the device. Each device should have a port, a name and a log_dir to store temporary information (e.g. models, data, results).

        :param port: Port to which the device is connected
        :param name: Name of the device
        :param log_dir: Directory where to store temporary information
        :param kwargs: Other parameters
        """
        self.port = port
        self.name = name
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        else: # generate logdir based on current date and time
            self.log_dir = os.path.join(os.getcwd(), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.is_ready_for_inference = False

    # SYSTEM PREPARATION
    @abstractmethod
    def __prepare_model__(self, model: tf.keras.models.Model, **kwargs):
        """
        Prepare the model by following all the opitimization steps required by the device, and stores it.

        :param model: a Keras model
        :param kwargs: other parameters usefule for the preparation of the model
        """
        raise NotImplementedError()

    @abstractmethod
    def __generate_inference_code__(self):
        """
        Generate the code or the executable for inference on the device.

        :return: path to executable or directory containing the code
        """
        raise NotImplementedError()

    @abstractmethod
    def __deploy_model__(self):
        """
        Deploy the model on the device.

        :raises Exception: if an error occurs during deployment
        """
        raise NotImplementedError()

    @abstractmethod
    def __deploy_code__(self):
        """
        Deploy the executable code for inference on the device.

        :raises Exception: if an error occurs during deployment of the code
        """
        raise NotImplementedError()

    def prepare_for_inference(self, model: tf.keras.models.Model, **kwargs):
        """
        Prepare the device for inference. This function should be called before any inference.

        :param model: model to deploy on the device
        :param kwargs: other parameters relevant for the preparation of the device
        """
        self.__prepare_model__(model, **kwargs)
        self.__generate_inference_code__()
        self.__deploy_model__()
        self.__deploy_code__()
        self.is_ready_for_inference = True

    # INFERENCE
    @abstractmethod
    def __prepare_data__(self, input_x, **kwargs):
        """
        Prepare the data for inference on the device.

        :param input_x: data to prepare for the deployment
        :param kwargs: other parameters
        :return: prepared data
        """
        raise NotImplementedError()

    def __subsample_data__(self, input_x, batch_size: int, batch_index: int):
        """
        Subsample the data to fit the device memory.

        :param input_x: data to subsample
        :param batch_size: batch size
        :param batch_index: index of the batch
        :return: a batch of the subsampled data
        """
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        return input_x[start_index:end_index]

    @abstractmethod
    def __transfer_data_to_memory__(self, input_x):
        """
        Transfer the data to the device memory.
        This data will be read by the inference script or executable.

        :param input_x: data to transfer
        :raise Exception: if an error occurs during the transfer
        """
        raise NotImplementedError()

    @abstractmethod
    def __run_inference__(self, profile: bool = True):
        """
        Starts the inference on the device, by executing the code on device or the executable.

        :param profile: if true, the inference is profiled
        """
        raise NotImplementedError()

    @abstractmethod
    def __read_inference_results__(self):
        """
        Read the inference results from the device memory, and transfer them back to the workstation.

        :return: inference results, profiler object
        """
        raise NotImplementedError()

    def infer(self, input_x, batch_size: int, **kwargs):
        """
        Run inference on the device and measures associated performance metrics.

        :param input_x: input data to infer
        :param batch_size: batch size
        :param kwargs: other parameters
        :return: tuple: (inference_results: np.array, profiler: Profiler)
        :raises DeviceNotReadyException: if the device is not ready for inference.
        """
        if not self.is_ready_for_inference:
            raise DeviceNotReadyException(
                "The device is not ready for inference. Please call prepare_for_inference() first.")

        input_x = self.__prepare_data__(input_x, **kwargs)

        all_predictions = []
        all_profilers = []

        for i in range(len(input_x) // batch_size):
            batch = self.__subsample_data__(input_x, batch_size, i)
            self.__transfer_data_to_memory__(batch)
            self.__run_inference__(profile=True)
            predictions, profiler = self.__read_inference_results__()
            all_predictions.append(predictions)
            all_profilers.append(profiler)

        # merge predictions
        y_pred = np.concatenate(all_predictions, axis=0)
        profiler = Profiler.merge(all_profilers)

        return y_pred, profiler

    def predict(self, input_x, batch_size, **kwargs):
        """
        Run inference on the device and measures associated performance metrics.

        :param input_x: input data to infer
        :param batch_size: batch size
        :param kwargs: other parameters
        :return: the inference results
        """
        input_x = self.__prepare_data__(input_x, **kwargs)

        if not self.is_ready_for_inference:
            raise DeviceNotReadyException(
                "The device is not ready for inference. Please call prepare_for_inference() first.")

        all_predictions = []

        for i in range(len(input_x) // batch_size):
            batch = self.__subsample_data__(input_x, batch_size, i)
            self.__transfer_data_to_memory__(batch)
            self.__run_inference__(profile=False)
            predictions, _ = self.__read_inference_results__()
            all_predictions.append(predictions)

        # merge predictions
        predictions = np.concatenate(all_predictions, axis=0)

        return predictions

    # Helper functions
    @abstractmethod
    def is_alive(self, timeout: int = 20) -> bool:
        """
        Check if the device is alive

        :param timeout: timeout in seconds
        :return: Return true if the device is alive (connected), false otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        """
        Representats the device as a string.
        :return: String representation of the device.
        """
        raise NotImplementedError()
