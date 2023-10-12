#!/user/bin/env python

"""
Author: Romain Gaulier
Email: romain.gaulier@csem.ch
Copyright: CSEM, 2023
Creation: 21.04.23
Description: Interface of task
"""
from abc import abstractmethod
from typing import Union
import numpy as np


class Task:
    """
    Class representation of task
    """

    def __init__(self):
        """
        Constructor of the task.
        """
        self.train_data = np.array([])
        self.test_data = np.array([])
        self.validation_data = np.array([])

        self.metrics = None

    @abstractmethod
    def get_train_data(self):
        """
        Returns the train data used for the task.

        :return: X and Y, where X is the input(s) data and Y is the output(s) data.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        """
        Returns the test data used for the task.
        :return: X and Y, where X is the input(s) data and Y is the output(s) data.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_validation_data(self):
        """
        Returns the validation data used for the task.
        :return: X and Y, where X is the input(s) data and Y is the output(s) data.
        """
        raise NotImplementedError()

    def get_data(self) -> Union[list, tuple, np.array]:

        """
        Returns the data used for the task.
        :return: if validation_data is None, returns train_data, test_data. Otherwise, returns train_data, validation_data, test_data.
        """
        if self.validation_data is not None:
            return self.get_train_data(), self.get_validation_data(), self.get_test_data(),
        else:
            return self.get_train_data(), self.get_test_data()

    def evaluate(self, y: np.array, y_pred: np.array):
        """
        Evaluate the model on the given data, using the metrics defined in the task.
        :param y: true labels (could be self.y_test, self.y_train, self.y_validation)
        :param y_pred: predicted labels
        :return:
        """
        res = {}
        for k in self.metrics.keys():
            self.metrics[k].reset_states()
            self.metrics[k].update_state(y, y_pred, sample_weight=None)
            res[k] = self.metrics[k].result().numpy()

    @abstractmethod
    def get_metrics_info(self) -> dict:
        """
        Returns a dictionary containing the metrics used for the task. The keys should be the name of the metrics, and the values
        should be a description of the metrics.
        :return: Dictionary containing the metrics used for the task.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_metrics(self):
        """
        Returns a dictionary containing the metrics used for the task. The keys should be the name of the metrics, and the values
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the task
        :return:
        """
        raise NotImplementedError()
