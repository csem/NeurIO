========================
Block-based Benchmark
========================

Edge devices are usually very constrained in terms of resources. Some devices may execute some layer faster than other,
while some devices might not support all layers. This makes it difficult to compare the performance of different devices.

To solve this problem, NeurIO provides a block-based benchmarking framework. This framework allows to compare the performance
of different devices by measuring the time it takes to execute each layer of the model.

.. _layer-based-benchmarking:

Layer-based benchmarking
------------------------

The basic blocks are consist in popular layers used in neural networks. These blocks are:

- Convolutional 1D layer
- Convolutional 2D layer
- Depthwise 2D convolutional layer
- Convolutional 3D layer
- Fully connected layer
- Pooling layer
- Activation layer
- Batch normalization layer
- Reshape layer
- Padding

The available configurations for each block are as follow

+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Block            | Config-1              | Config-2              | Config-3              | Config-4              |
+==================+=======================+=======================+=======================+=======================+
| Convolutional 1D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Convolutional 2D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Convolutional 2D | filters 32, 1x1       | filters 32, 3x3       | filters 32, 5x5      | filters 64, 7x7        |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Depthwise 2D conv| filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Convolutional 3D | filters 32, stride 1  | filters 32, stride 2  | filters 64, stride 1  | filters 64, stride 2  |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Fully connected  | 128-32 neurons        | 256-64 neurons        | 512-128 neurons       | 1024-256 neurons      |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| AvgPooling1D     | 2, stride 1           | 2, stride 2           | 4, stride 2           | 7, stride 1           |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| AvgPooling2D     | 2x2, stride 1         | 2x2, stride 2         | 4x4, stride 2         | 7x7, stride 1         |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| MaxPooling1D     | 2, stride 1           | 2, stride 2           | 4, stride 2           | 7, stride 1           |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| MaxPooling2D     | 2x2, stride 1         | 2x2, stride 2         | 4x4, stride 2         | 7x7, stride 1         |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Activation       | ReLU                  | Sigmoid               | ReLU6                 | Tanh                  |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Batch norm       | 1D                    | 2D                    | 3D                    | 4D                    |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Reshape          | 1D                    | 2D                    | 3D                    | 4D                    |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Padding 1D       | Zero                  | Constant              | Reflect               | Symmetric             |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Padding 2D       | Zero                  | Constant              | Reflect               | Symmetric             |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+


Usage
_____

To use the block-based benchmarking framework, you need to create a `LayerBasedBenchmark` object. This object will be used to
benchmark the different blocks. The `LayerBasedBenchmark` object can be created as follow:

.. code-block:: python

    from neurio.benchmark import LayerBasedBenchmark

    benchmark = LayerBasedBenchmark()
    device = ... # Create a device object
    results = benchmark.run_on(device)

Once the `LayerBasedBenchmark` object is created, you can run the benchmark on a connected device as follow.
The `run_on` method will run all the benchmarks on the device and return a `BenchmarkResult` object.
This object contains the results of the benchmark.

.. _cell-based-benchmarking:

Cell-based benchmarking
-----------------------

Cells are blocks that are composed of multiple layers. Usually, they are building blocks of popular topologies
used in neural networks. These cells can be:

- Skip connection (Concatenate and Add)
- ResNet block
- DenseNet block
- Inception block
- MobileNet block
- Classifier block

The available configurations for each block are as follow:

+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Block            | Config-1              | Config-2              | Config-3              | Config-4              |
+==================+=======================+=======================+=======================+=======================+
| Skip connection  | Concatenate-2         | Add-2                 | Concatenate-4         | Add-4                 |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| ResNet block     | ConvBlock 64 filters  | IID Block 64 filters  | ConvBlock 128 filters | IID Block 128 filters |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| DenseNet block   | Conv1D 32 filters     | Conv1D 64 filters     | Conv2D 32 filters     | Conv2D 64 filters     |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Inception block  | Naive-1D              | With reduction-1D     | Naive-2D              | With reduction-2D     |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| MobileNet block  | Block 32 filters      | Block 64 filters      | Block 128 filters     | Block 256 filters     |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Classifier block | Avg2D-Flatten-Dense10 | GlAvg-Flatten-Dense10 |          -            |         -             |
+------------------+-----------------------+-----------------------+-----------------------+-----------------------+

Usage
_____

To use the composed block-based benchmarking framework, you need to create a `CellBasedBenchmark` object.
This object will be used to benchmark the different blocks. The `CellBasedBenchmark` object can be used as follow:

.. code-block:: python

    from neurio.benchmark import CellBasedBenchmark

    benchmark = CellBasedBenchmark()
    device = ... # Create a device object
    results = benchmark.run_on(device)

Once the `CellBasedBenchmark` object is created, you can run the benchmark on a connected device as follow.
The `run_on` method will run all the benchmarks on the device and return a `BenchmarkResult` object.
This object contains the results of the benchmark.
