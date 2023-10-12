# Getting Started

Welcome to NeurIO, an open-source library for deploying machine learning models on embedded devices. In this section, we will guide you through the process of getting started with the library and performing your first basic inference task on a device.


## Quickstart

Let’s start with a quick example. We will use the MNIST dataset and a simple convolutional neural network to classify handwritten digits. We will then deploy the model on a device and perform inference an ST Microleectronic platform.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score

from neurio.devices.virtual.canaan.kendryte import K210Virtual

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

input_shape = x_train.shape[1:]

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

n = 100 # number of samples to test

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
    
print("Scores:")
print("Original model: {:.2f}".format(score_original))
print("Device: {:.2f}".format(score_device))

print("Profiling information:")
print(profiler)
```

This example demonstrates the basic workflow:

- Initialize an edge device (in this case, a ).
- Choose a task (image classification).
- Choose a model (ResNet-50).
- Deploy the selected model on the edge device.
- Load an image for inference.
- Perform inference using the specified task.
- Display the inference result.

This is just a simple introduction to NeurIO. In the following sections, we’ll delve deeper into each step and explore more advanced features and customization options.

Now that you’ve successfully run your first inference task, let’s dive into more details about the supported edge devices, tasks, and models in the subsequent sections. Next Steps

- Check out the Supported Edge Devices section to learn more about the devices compatible with NeurIO.
- Explore the Choosing a Task section to understand the available machine learning tasks.
- Discover the Choosing a Benchmark section to find out about the pre-trained models on several tasks that you can deploy.
- Dive into the Deployment section to get a deeper understanding of the deployment process.

Congratulations! You’ve taken your first steps with NeurIO. Feel free to reach out in case you have any questions or encounter any issues along the way.