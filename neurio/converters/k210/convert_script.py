#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 13.07.23
Description: TODO
"""

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import argparse
import os

from neurio.devices.physical.canaan import K210VirtualDevice

# write main taking "model path" as argument:

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_path", type=str, default="model.h5")
    args = argparse.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    if not os.path.exists(args.model_path):
        raise ValueError("Model path does not exist")
    if not args.model_path.endswith(".h5"):
        raise ValueError("Model path must be a .h5 file")

    model = load_model(args.model_path, compile=False)

    model.summary()

    head_model = Sequential([Input(shape=(2,1,1)), Flatten(), model.layers[1], model.layers[2], model.layers[3]])
    head_model.summary()

    model = head_model
    # generate random input of the right shape
    model_input_shape = model.input_shape
    model_input_shape = tuple([10] + list(model_input_shape[1:]))
    model_input = np.asarray(np.random.rand(*model_input_shape)*255).astype(np.uint8)



    # run model
    model_output = model.predict(model_input)

    # use VirtualDevice to get output
    device = K210VirtualDevice(port="virtual", log_dir=os.path.dirname(args.model_path))
    device.prepare_for_inference(model)
    device_output = device.predict(model_input)

    # compare outputs
    print("Model output: ", model_output)
    print("Device output: ", np.asarray([device_output.pred]))

    print("Error: ", np.sum(np.abs(model_output - np.asarray([device_output.pred]))))





