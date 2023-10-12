=======================
Task-based Benchmarking
=======================

Edge devices are usually very constrained in terms of resources.
Therefore, we have defined a few tasks that can be used to benchmark the performance of a device.

Available Tasks
---------------

This means that they can only perform a limited set of tasks, and might be optimized for a specific task.
For example, a device might be optimized for image classification, but not for sound recognition.
NeurIO currently supports the following tasks:

+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| Task Name  | Dataset   | Domain | Category       | Main Metric | Description                                                       |
+============+===========+========+================+=============+===================================================================+
| `mnist`    | MNIST     | Image  | Classification | Accuracy    | Digit classification on 10 classes                                |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| `cifar10`  | CIFAR-10  | Image  | Classification | Accuracy    | Object classification on 10 classes                               |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| `cifar100` | CIFAR-100 | Image  | Classification | Accuracy    | Object classification on 100 classes                              |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| `imagenet` | ImageNet  | Image  | Classification | Accuracy    | Object classification on 1000 classes                             |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| `N-MNIST`  | N-MNIST   | Events | Classification | Accuracy    | Digit classification using N-MNIST events                         |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+
| `SHD`      | SHD       | Events | Classification | Accuracy    | Audio digit classification using Spiking Heidelberg Digits events |
+------------+-----------+--------+----------------+-------------+-------------------------------------------------------------------+

Loading a task in NeurIO
------------------------

To choose a task, you can load the task from the ``neurio.tasks`` module:

.. code-block:: python

    from neurio.tasks import mnist

    task = MNIST()

    # get the data
    (x_train, y_train), (x_test, y_test) = task.get_data()

    # define a model an perform inference on device
    model = ...
    device = ...
    device.prepare_for_inference(model)
    y_pred, profiler = device.predict(x_test[:12], batch_size=4, return_stats=True)

    # evaluate the results
    metrics = task.evaluate(y_test[:12], y_pred)
    print(metrics)


Defining a new task
-------------------

To define a new task, you need to create a new class that inherits from the `Task` class.
This class must implement the following methods:

- `get_data`: returns the dataset used for the task.
- `evaluate`: evaluates the results of the inference and returns a dictionary of metrics.
- `get_metrics_info`: returns a description of the metrics used to evaluate the task.

For example, this is the implementation of the MNIST classification task:

.. code-block:: python

    #!/user/bin/env python

    import numpy as np
    import tensorflow as tf

    from neurio.tasks.task import Task


    class MNISTClassification(Task):
        def __init__(self):
            super().__init__()
            self.x = None
            self.y = None
            self.metric = {"CE": tf.keras.metrics.CategoricalCrossentropy(),
                           "Accuracy": tf.keras.metrics.CategoricalAccuracy()}

            self.dataset = tf.keras.datasets.mnist

            self.prepare_data()

        def get_data(self, preprocess=False):
            (x_train, y_train), (x_test, y_test) = self.dataset.load_data()

            x_train = np.repeat(np.expand_dims(x_train, axis=3), 3, axis=3)
            x_test = np.repeat(np.expand_dims(x_test, axis=3), 3, axis=3)

            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

            # Normalize value
            if preprocess:
                x_train = (x_train / 255.0).astype(np.float32)
                x_test = (x_test / 255.0).astype(np.float32)

            return (x_train, y_train), (x_test, y_test)

        def evaluate(self, y_train, y_pred):
            res = {}
            for k in self.metric.keys():
                self.metric[k].reset_states()
                self.metric[k].update_state(y_train, y_pred, sample_weight=None)
                res[k] = self.metric[k].result().numpy()
            return res

        def get_metrics_info(self):
            return {"CE": "Categorical Cross Entropy",
                    "Accuracy": "Accuracy over 10 classes"}