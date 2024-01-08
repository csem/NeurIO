#!/user/bin/env python

"""
Author: Romain Gaulier
Email: romain.gaulier@csem.ch
Copyright: CSEM, 2023
Creation: 26.04.23
Description: Pipeline for CIFAR10 dataset
"""

import numpy as np
import tensorflow as tf

from neurio.tasks.task import Task


class CIFAR10Classification(Task):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None
        self.metric = tf.keras.metrics.CategoricalCrossentropy()
        self.dataset = tf.keras.datasets.cifar10

        self.prepare_data()

    def preprocess(self):
        (x_train, y_train), (x_test, y_test) = self.dataset.load_data()

        x_train = np.repeat(np.expand_dims(x_train, axis=3), 3, axis=3)
        x_test = np.repeat(np.expand_dims(x_test, axis=3), 3, axis=3)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        # Normalize value
        x_train = (x_train / 255.0).astype(np.float32)
        x_test = (x_test / 255.0).astype(np.float32)

        self.x = x_train.numpy()
        self.y = y_train.numpy()

    def evaluate(self, y_train, y_pred):
        self.metric.update_state(y_train, y_pred, sample_weight=None)
        return self.metric.result().numpy()
