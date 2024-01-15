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
    def __prepare_model__(self, model, **kwargs):
        """
        Prepare the model by following all the opitimization steps required by the device, and stores it.

        :param model: a Keras model
        :param kwargs: other parameters usefule for the preparation of the model
        """
        pass

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
