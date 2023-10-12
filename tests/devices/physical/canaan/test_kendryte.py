#!/user/bin/env python

import pytest
import tensorflow as tf
import os

from neurio.devices.physical.canaan import K210

def test_deployment_pipeline():
    #TODO mock connection to device
    device = K210(port="/dev/tty.usbserial-14240", verbose=1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]

    if not os.path.exists("cifar10.h5"):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics="accuracy")
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        model.summary()
        model.save("cifar10.h5")
    else:
        model = tf.keras.models.load_model("cifar10.h5")

    device.prepare_for_inference(model)
    input_x = x_train[:4]
    predictions, profiler = device.predict(input_x, batch_size=2)