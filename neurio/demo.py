#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 17.04.23
Description: TODO
"""
import tensorflow as tf
from neurio.devices import K210Device
import time
import numpy as np
import os

def main_demo():

    model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=[28, 28, 3]),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=10, activation="softmax")
            ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy')  # compile the model

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    n = 128
    x_train = np.repeat(np.expand_dims(x_train, axis=3), 3, axis=3)[:n]
    x_test = np.repeat(np.expand_dims(x_test, axis=3), 3, axis=3)[:n]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)[:n]
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)[:n]

    x_train = x_train * 255
    x_test = x_test * 255

    model.fit(x=x_train, y=y_train, epochs=1)  # train the model

    log_dir = os.path.join("./logs", str(time.time()))
    device = K210Device(port="/dev/tty.usbserial-14420", name="K210", log_dir=log_dir)

    device.prepare_for_inference(model, options={"calibration_dataset": np.asarray(x_train[:128], dtype=np.uint8)})

    y_pred = device.predict(x_test[:12], batch_size=4)

if __name__ == "__main__":
    main_demo()