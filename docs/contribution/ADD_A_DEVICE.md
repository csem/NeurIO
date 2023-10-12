# Adding a physical device in NeurIO

NeurIO is a powerful library designed to simplify the deployment of machine learning models onto hardware devices.
Adding a new device to the NeurIO library allows you to leverage its capabilities for efficient model deployment. Here's
a step-by-step guide to adding a device to the NeurIO library:

## Step 1: Download NeurIO and create a new branch

Download the NeurIO from Github:

```shell
git clone https://github.com/csem/neurio.git
```

Create a branch for your device (replace the value of *device_name* by your device):

```sh
git checkout -b development/feat/{device_name}
```

## Step 2:  Create a new Device Class

Inside the NeurIO repository, navigate to the neurio/devices directory and list all constructors.

```sh
cd neurio/devices/physical & ls
```

If you don't see your constructor listed, create a new directory for your platform (replace {constructor} with your
platform's constructor name):

```sh
mkdir {constructor}
```

Inside the newly created {constructor} directory, create a Python file named {device_family}.py (replace {device_family} with
your platforms device model name).
Define your platform's device class inside it.
Ensure that it inherits from the Device class as specified in your provided code snippet.
Customize the class according to your platform's needs and functionalities.

```python
# Example structure in {my_device}.py
from neurio.devices.device import Device


class MyDevice(Device):
# Implement all abstract methods inherited from and your platform-specific methods and attributes here

```

## Step 3: Create the documentation for the device

After the implementation of the device class, you must create the documentation to ensure that the device is properly
documented.
The documentation is written in Markdown and is located in the docs/supported device directory.

The documentation should contain the followning sections:

1. **Introduction**:
   A short introduction to the device, including the manufacturer, the model name, and a short description of the
   device.
2. **Installation**: A section to explain how to install the required tools for the deployment pipleine.
   A link to the documentation of each external tool should be provided.

3**Deployment Pipeline**: A detailed figure of the deployment pipeline of the device.
The deployment pipeline should explain each step of the deployment (System Preparation and Prediction). See the
structure
of the [Generic Deployment Pipeline](../basics/generic_deployment.md).

To help with the process, we provide a [template](../contribution/template.md) that you can use to create the
documentation for your device.

## Step 4: Create a functional test for your device

Each device in NeurIO has a functional test that ensures that the device is working properly.
Whenever a new device is added to NeurIO, one must create a functional test for it.
The functional test is a Python script that performs a simple inference task on the device.

To create a test, navigate to the tests/functional directory.
If it does not yet exists, create a folder from the name of the chip constructor (e.g, `stm` for ST Microelectronics),
and create a new Python file named
`test_{my_device}.py` (replace {my_device} with your platform's device model name).

Inside the test file, create a new test class named `Test{MyDevice}` (replace {MyDevice} with your platform's device
model name).

```python
    # Example structure in test_{my_device}.py

from neurio.devices.physical.{constructor}.{device_family} import {MyDevice}
from neurio.tests.functional import FunctionalTest

class Test{MyDevice}(FunctionalTest):
    # Implement your test here
```

## Step 5: Submit a pull request

Once you have completed the previous steps, you can submit a pull request to the NeurIO repository to merge
on the `development` branch.

