======================
Model-based benchmarks
======================

Model-based benchmarks rely on architectures of neural networks to measure the performance.
The weights of the architectures are either trained on a dataset or generated randomly.

In NeurIO, we provide a set of model-based benchmarks that are based on popular neural network architectures.
The benchmarks are designed to be simple to use and to be easily extensible.

This is the list of available benchmarks that are based on models.

+----------------+----------+-----------------------------------------------------------------+
| Benchmark Name | Weights  | Description                                                     |
+================+==========+=================================================================+
| `layer-based`  |          | This benchmark measures the performance                         |
|                | Random   | of a device on single layers.                                   |
|                |          | See :ref:`layer-based-benchmarking` for more details.           |
+----------------+----------+-----------------------------------------------------------------+
| `cell-based`   |          | This benchmark measures the performance of a device on blocks   |
|                | Random   | composed of multiple layers (cells), which are building         |
|                |          | blocks of a popular neural network.                             |
|                |          | See :ref:`cell-based-benchmarking` for more details.            |
+----------------+----------+-----------------------------------------------------------------+
| `tiny_ml_perf` |          | [SOON] This benchmark is based on the [tinyMLPerf]() benchmark. |
|                |Pretrained| It is a simple benchmark that measures the performance          |
|                |          | of a device on a simple CNN model.                              |
+----------------+----------+-----------------------------------------------------------------+

.. note::
    The `tiny_ml_perf` benchmark is not yet available.

See more details:

.. toctree::
    :maxdepth: 2

    block_based_benchmarking
    tiny_ml_perf