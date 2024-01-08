#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 08.05.23
Description: TODO
"""

import nncase
import os
import argparse

import numpy as np
import tensorflow as tf


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def convert(args):
    # model
    filename = args.filename
    model = filename

    print("\n\n=== Model description ===")
    print("Detected format", filename.split('.')[-1])
    if "tflite" in filename:
        name = filename.split('.tflite')[0]
        # read TFLITE model and extract input shape
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

        input_shapes = [inp['shape'] for inp in input_details]
        # check that the last dimension is channel
        print("Transformation on the input shapes:")
        for shape in input_shapes:
            is_NHWC = ((shape[-1] < shape[-2]) & (shape[-1] < shape[-3]))

            if is_NHWC and args.layout == 'NCHW':
                # if NHWC, change to NCHW, by swapping the last dimension and sending to index 1
                old_shape = shape.tolist()
                shape[-1], shape[1] = shape[1], shape[-1]
                print("\tNHWC -> NCHW : {} -> {}".format(old_shape, shape))

        if len(input_shapes) == 1:
            input_shapes = input_shapes[0]
        args.input_shape = input_shapes

    elif "onnx" in filename:
        name = filename.split('.onnx')[0]
    else:
        raise Exception("Only TFLITE and ONNX are supported")
    target = 'k210'

    print("\n=== Compilation phase ===")

    # import
    model_content = read_model_file(model)
    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_shape = args.input_shape
    compile_options.input_layout = args.layout
    compile_options.output_layout = args.layout  # You can change this option to what you want.
    compile_options.model_layout = 'NHWC'
    compile_options.preprocess = True
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    if "tflite" in filename:
        compiler.import_tflite(model_content, import_options)
    else:
        compiler.import_onnx(model_content, compile_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    if "onnx" in filename:
        name = name + "-onnx"
    kmodel_file = name + ".kmodel"
    with open(kmodel_file, 'wb') as f:
        f.write(kmodel)

    # simulate result

    simulator = nncase.Simulator()
    simulator.load_model(kmodel)

    print("\n=== Evaluation ===")

    if args.reprensentative_dataset is not None:
        dataset = np.load(args.reprensentative_dataset)

        # dataset = tf.keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test) = dataset
        dataset = [(x_train, y_train), (x_test, y_test)]

        num_inputs = len(dataset[0])
        num_outputs = len(dataset[1])

        if len(num_inputs) > 3:
            print("Detected size of dataset is larger than 3, assuming it does not contain input size (1).")
            input_shapes = [dataset[0][:1].shape]
            num_inputs = 1
        if len(num_outputs) > 3:
            print("Detected size of dataset is larger than 3, assuming it does not contain output size (1).")
            output_shapes = [dataset[1][:1].shape]
            num_outputs = 1

        else:
            input_shapes = [x.shape for x in dataset[0]]
            output_shapes = [x.shape for x in dataset[1]]

        print("DETECTED SHAPES FROM DATASET")
        print("Input shapes: ", input_shapes)
        print("Output shapes: ", output_shapes)

        print("Simulating...")
        preds = []
        for i in range(num_inputs):
            inputs = dataset[0][i]
            for i in range(len(inputs)):
                rt_x = nncase.RuntimeTensor.from_numpy(inputs[i])
                simulator.set_input_tensor(i, rt_x)
            simulator.run()
            pred = simulator.get_output_tensor(i).to_numpy()
            preds.append(pred)
        print("Done!")

        errors = []
        for i in range(num_outputs):
            outputs = dataset[1][i]
            pred = preds[i]
            errors.append(np.mean(np.abs(outputs - pred)))
        print("Mean errors:", errors)

        accuracies = []
        for i in range(num_outputs):
            labels = dataset[1][i].argmax(-1)
            pred = preds[i].argmax(-1)
            accuracies.append(np.mean(labels == pred))
        print("Accuracies:", accuracies)

    else:

        num_inputs = simulator.inputs_size
        for i in range(num_inputs):
            input_shape = simulator.get_input_tensor(i).shape
            x = np.random.random(input_shape)
            x = np.asarray(x, dtype=np.float32)
            rt_x = nncase.RuntimeTensor.from_numpy(x)
            simulator.set_input_tensor(i, rt_x)
            print("\tInput {} shape: {}".format(i, input_shape))
        simulator.run()
        print("\tSimulation ok.")
        print("\tDetected output")
        num_outputs = simulator.outputs_size
        for i in range(num_outputs):
            output_shape = simulator.get_output_tensor(i).to_numpy().shape
            print("\tOutput {} shape: {}".format(i, output_shape))

        print("Success! Model saved as: {}".format(kmodel_file))


if __name__ == '__main__':
    # Arguments for reading filename
    parser = argparse.ArgumentParser(description='Convert tflite to kmodel')
    parser.add_argument('--filename', type=str, default='model.tflite', help='Filename of tflite model')
    parser.add_argument('--reprensentative_dataset', type=str, default=None,
                        help='Numpy file of representative dataset')
    parser.add_argument('--input_shape', help='Input shape of the kmodel, can be passed as list of int', nargs='+',
                        type=int, default=None)
    parser.add_argument("--layout", type=str, default="NCHW", help="Input format of the kmodel")
    args = parser.parse_args()

    convert(args)
