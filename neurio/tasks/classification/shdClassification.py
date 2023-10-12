#!/user/bin/env python

"""
Author: Romain Gaulier
Email: romain.gaulier@csem.ch
Copyright: CSEM, 2023
Creation: 26.04.23
Description: Pipeline for SHD dataset
"""

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
#import snntorch as snn
#from snntorch.spikevision import spikedata

import sys
import os

from neurio.tasks.task import Task


class SHDClassification(Task):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None
        self.metric = tf.keras.metrics.CategoricalCrossentropy()
        self.train_ds = spikedata.SHD("dataset/shd", train=True)
        self.test_ds = spikedata.SHD("dataset/shd", train=False)

        self.prepare_data()

    def prepare_data(self):
        # Convert the spike-based data to regular image arrays
        train_images = [self.train_ds[i][0] for i in range(len(self.train_ds))]
        train_labels = [self.train_ds[i][1] for i in range(len(self.train_ds))]
        test_images = [self.test_ds[i][0] for i in range(len(self.test_ds))]
        test_labels = [self.test_ds[i][1] for i in range(len(self.test_ds))]

        # Convert the image arrays to TensorFlow datasets
        train_images_ds = tf.data.Dataset.from_tensor_slices(train_images)
        train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)
        test_images_ds = tf.data.Dataset.from_tensor_slices(test_images)
        test_labels_ds = tf.data.Dataset.from_tensor_slices(test_labels)

        # Combine the image and label datasets
        train_ds = tf.data.Dataset.zip((train_images_ds, train_labels_ds))
        test_ds = tf.data.Dataset.zip((test_images_ds, test_labels_ds))

        # Shuffle and batch the datasets
        batch_size = 64
        x_train = train_ds.shuffle(len(train_ds)).batch(batch_size)
        x_test = test_ds.batch(batch_size)

        # Convert labels to one-hot encoding
        num_classes = 10  # TODO: Set the value
        y_train = x_train.map(lambda x, y: (x, self.metric(y, num_classes)))
        y_test = x_test.map(lambda x, y: (x, self.metric(y, num_classes)))

        x_train = (x_train / 255.0).astype(np.float32)
        x_test = (x_test / 255.0).astype(np.float32)

        self.x = x_train.numpy()
        # Not sure?
        self.y = y_train.numpy()

    def evaluate(self, y_train, y_pred):
        self.metric.update_state(y_train, y_pred, sample_weight=None)
        return self.metric.result().numpy()
