#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 06.02.23
Description: Entry point for benchmarking
"""
import tensorflow as tf
from .stm import STMDevice
from neurio.tasks import CIFAR10Classification
from neurio.tasks import CIFAR10ClassificationMAE
from neurio.tasks import ImageNetClassification
from neurio.tasks import NMNISTClassification
from neurio.tasks import SHDClassification
from neurio.tasks import MNISTClassification
from .tflite import TFLiteDevice
from common import Benchmarking
from neurio.devices.physical.stmicroelectronics import STMDevice
from neurio.devices.physical.canaan import K210Device
from neurio.devices.virtual.tensorflow.tflite import TFLiteDevice
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import datetime
import os

# Supported devices for ST family
st_devices = [
    'STM32L4R9'.lower(),
    'NUCLEO-H723ZG'.lower(),
    'NUCLEO-L552ZE-Q'.lower(),
    'STM32H747I-DISCO'.lower(),
]

tf_devices = [
    "TFLite".lower(),
]

canaan_devices = [
    'K210'.lower(),
]
# Create a new pipeline : List of supported devices
new_family_devices = [

]

VERSION = [0, 1, 0]

# Benchmark function
def benchmark(device: str, task_name: str, model: tf.keras.Model, compression: str, validation_inputs: str, validation_outputs: str,
              batch_size: int, is_quantized: bool, output: str) -> Benchmarking:
    print(device, task_name, model, compression, validation_inputs, validation_outputs, batch_size, is_quantized, output)

    device = device.lower()
    y_pred = None

    # ST pipeline
    if device in st_devices:
        device = STMDevice(port="/dev/ttyACM0", name=device, log_dir=output,
                           options={'compression': compression, 'is_quantized': is_quantized})

    elif device in tf_devices:
        device = TFLiteDevice(port="/dev/ttyACM0", name=device, log_dir=output,
                              options={'compression': compression, 'is_quantized': is_quantized})
    elif device in canaan_devices:
        device = K210Device(port="/dev/ttyACM0", name=device, log_dir=output,
                              options={'compression': compression, 'is_quantized': is_quantized})
    else:
        raise ValueError('Device not supported. Refers to de documentation: Add a new device to package')

    tasks = {
        "cifar10": CIFAR10Classification(),
        "cifar10MAE": CIFAR10ClassificationMAE(),
        "imageNet": ImageNetClassification(),
        "mnist": MNISTClassification(),
        "nmnist": NMNISTClassification(),
        "shd": SHDClassification()
    }
    task = tasks.get(task_name)
    if task is None:
        raise NameError("Unknown task: " + task_name)

    # perform prediction
    device.prepare_for_inference(model)
    profiler = device.profiler
    y_pred, profiler = device.predict(task.x)
    score = task.evaluate(task.y, y_pred)

    # Calculate metrics related to predictions
    # results = profiler
    if validation_outputs is not None:
        refs = np.array(validation_outputs).argmax(axis=1)
        predictions = np.array(y_pred).squeeze().argmax(axis=1)
        accuracy = accuracy_score(refs, predictions)
        balanced_accuracy = balanced_accuracy_score(refs, predictions)
        profiler.accuracy = accuracy
        profiler.balanced_accuracy = balanced_accuracy

    # with open(os.path.join(output, "{}.json".format(datetime.now())), "w") as f:
    #     json.dump(results, f, cls=NpEncoder)
    profiler.save_json_data(os.path.join(output, "{}.json".format(datetime.now())))

    return Benchmarking(profiler)


# Device function
def devices():
    for device in st_devices:
        print(device)

    # Create a new pipeline: print supported devices
    for device in tf_devices:
        print(device)
