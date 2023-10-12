#!/user/bin/env python

import pytest
import tensorflow as tf
import os
from sklearn.metrics import balanced_accuracy_score

from neurio.devices.virtual.canaan import K210Virtual


def test_deployment_pipeline():
    # set seed
    tf.random.set_seed(0)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

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

        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics="accuracy")
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        model.summary()
        model.save("cifar10.h5")
    else:
        model = tf.keras.models.load_model("cifar10.h5")

    n = 100  # number of samples to test

    # perform inference on the original model
    y_pred_model = model.predict(x_train[:n])
    y_pred_model = tf.argmax(y_pred_model, axis=1)

    # Perform inference on the virtual device
    device = K210Virtual(verbose=1, log_dir="/tmp/k210_virtual")
    device.prepare_for_inference(model)
    input_x = x_train[:n]
    y_pred_device, profiler = device.infer(input_x, batch_size=2)
    y_pred_device = tf.argmax(y_pred_device, axis=1)

    y_true = tf.argmax(y_train[:n], axis=1)
    score_original = balanced_accuracy_score(y_true, y_pred_model)
    score_device = balanced_accuracy_score(y_true, y_pred_device)

    assert abs(score_original - score_device) < 0.05  # less than 5% difference

    # assert profiler is complete
    assert profiler["model"]["framework"] == "tensorflow"
    assert profiler["model"]["model_datetime"] is not None
    assert profiler["model"]["compile_datetime"] is not None

    assert profiler["inference"]["batch_size"] == 2
    assert len(profiler["inference"]["model_load_time"]) == n
    assert len(profiler["inference"]["preprocess_time"]) == n
    assert len(profiler["inference"]["inference_time"]) == n
    assert len(profiler["inference"]["readout_time"]) == n
