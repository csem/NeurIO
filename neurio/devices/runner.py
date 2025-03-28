#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 15.01.2024
Description: TODO
"""
import numpy as np
from neurio.devices.device import Device
from neurio.benchmarking.profiler import Profiler
from neurio.exceptions import DeviceNotReadyException
from tqdm import tqdm


class Runner:

    def __init__(self, device: Device, callbacks: list = []):
        self.device = device
        self.callbacks = callbacks
        self.phase = None

    def __on_phase_start__(self):
        for c in self.callbacks:
            c.__on_phase_start__(self.phase)

    def __on_phase_stop__(self):
        for c in self.callbacks:
            c.__on_phase_stop__(self.phase)

    def prepare_for_inference(self, model, **kwargs):
        """
        Prepare the device for inference. This function should be called before any inference.

        :param model: model to deploy on the device
        :param kwargs: other parameters relevant for the preparation of the device
        """
        self.phase = "prepare_for_inference"
        self.__on_phase_start__()
        self.device.__prepare_model__(model, **kwargs)
        self.__on_phase_stop__()

        self.phase = "generate_inference_code"
        self.__on_phase_start__()
        self.device.__generate_inference_code__()
        self.__on_phase_stop__()

        self.phase = "deploy_model"
        self.__on_phase_start__()
        self.device.__deploy_model__()
        self.__on_phase_stop__()

        self.phase = "deploy_code"
        self.__on_phase_start__()
        self.device.__deploy_code__()
        self.__on_phase_stop__()

        self.device.is_ready_for_inference = True

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

    def infer(self, input_x, batch_size: int, profile=True, **kwargs):
        """
        Run inference on the device and measures associated performance metrics.

        :param input_x: input data to infer
        :param batch_size: batch size
        :param kwargs: other parameters
        :return: tuple: (inference_results: np.array, profiler: Profiler)
        :raises DeviceNotReadyException: if the device is not ready for inference.
        """
        if not self.device.is_ready_for_inference:
            raise DeviceNotReadyException(
                "The device is not ready for inference. Please call prepare_for_inference() first.")

        self.phase = "prepare_data"
        self.__on_phase_start__()
        input_x = self.device.__prepare_data__(input_x, **kwargs)
        self.__on_phase_stop__()

        all_predictions = []
        all_profilers = []

        for i in tqdm(range(len(input_x) // batch_size)):
            self.phase = "subsample_data"
            batch = self.__subsample_data__(input_x, batch_size, i)

            self.phase = "transfer_data_to_memory"
            self.__on_phase_start__()
            self.device.__transfer_data_to_memory__(batch)
            self.__on_phase_stop__()

            self.phase = "run_inference"
            self.__on_phase_start__()
            self.device.__run_inference__(profile=profile)
            self.__on_phase_stop__()

            self.phase = "read_inference_results"
            self.__on_phase_start__()
            predictions, profiler = self.device.__read_inference_results__()
            self.__on_phase_stop__()

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
        predictions, _ = self.infer(input_x, batch_size, profile=False, **kwargs)

        return predictions
