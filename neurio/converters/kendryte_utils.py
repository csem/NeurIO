#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 21.09.22
Description: Utils for the Kendryte conversion
"""

import os
import platform
import urllib.request
import warnings
import time
import contextlib
import numpy as np
import sys
import io

NCC_PATH = os.path.join(os.path.dirname(__file__), "ncc")
if not os.path.exists(NCC_PATH):
    os.makedirs(NCC_PATH, exist_ok=True)


github_link_prefix = {
    "2.0.0": "https://github.com/kendryte/nncase/releases/download/v2.0.0/nncase-2.0.0.20230602",
    "1.9.0": "https://github.com/kendryte/nncase/releases/download/v1.9.0/nncase-1.9.0.20230322",
    "1.8.0": "https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase-1.8.0.20220929",
    "1.7.1": "https://github.com/kendryte/nncase/releases/download/v1.7.1/nncase-1.7.1.20220701",
    "1.7.0": "https://github.com/kendryte/nncase/releases/download/v1.7.0/nncase-1.7.0.20220530",
    "1.6.0": "https://github.com/kendryte/nncase/releases/download/v1.6.0/nncase-1.6.0.20220505",
}

numpy_types =  {"float32": np.float32, "int8": np.int8, "uint8": np.uint8}
def install_ncc_python(version: str = "1.7.1", python: str = "38"):
    if "." in python:
        versions = python.split(".")
        python = versions[0] + versions[1]
    if "." not in version:
        # version of nncase should be in format 1.x.x
        raise Exception("Version of NNCase should be in format x.x.x")


    github_link = "https://github.com/kendryte/nncase/releases/download/v{}".format(version)

    os_platform = "linux"
    if platform.system() == "Darwin":
        os_platform = "macosx"

    if os_platform == "linux":
        file_download = github_link_prefix[version]+"-cp{}-cp{}-{}_2_24_x86_64.whl".format(python,
                                                                                                python, "manylinux")
    elif os_platform == "macosx":
        file_download = github_link_prefix[version]+"-cp{}-cp{}-{}_10_15_x86_64.whl".format(python,
                                                                                                 python, os_platform)
    else:
        raise Exception(platform.system(), "not supported for NNCase", version, "on Python", python)

    save_file = os.path.join(NCC_PATH, os.path.basename(file_download))
    if not os.path.exists(save_file):
        print("Downloading from", file_download)
        urllib.request.urlretrieve(file_download, save_file)
        os.system("python3 -m pip install {}".format(save_file))



@contextlib.contextmanager
def suppress_output(verbose=False):
    if verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
    else:
        yield

def install_ncc_shell(version: str = "0.2.0-beta4"):
    github_link = "https://github.com/kendryte/nncase/releases/download/v{}".format(version)
    # default linux
    os_platform = "linux"
    file_download = github_link + "/ncc_linux_x86_64.tar.xz"

    if platform.system() == "Darwin":
        os_platform = "macosx"
        file_download = github_link + "/ncc_osx_x86_64.dmg"

    NCC_EXEC_PATH = os.path.join(NCC_PATH, os_platform)
    os.makedirs(NCC_EXEC_PATH, exist_ok=True)
    save_file = os.path.join(NCC_EXEC_PATH, os.path.basename(file_download))
    if not os.path.exists(save_file) or not os.path.exists(os.path.join(NCC_PATH, "ncc")):
        print("Downloading from", file_download)
        urllib.request.urlretrieve(file_download, save_file)
        if save_file.endswith(".tar.xz"):
            os.system("tar -xf {} -C {}".format(save_file, NCC_PATH))
        elif save_file.endswith(".dmg"):
            print("Mounting dmg image...")
            os.system("hdiutil mount {}".format(save_file))
            os.system("cp -r /Volumes/bin/ncc {}".format(NCC_PATH))
            # unmount dmg
            print("Installation finished. Unmounting image.")
            os.system("hdiutil unmount /Volumes/bin")
        else:
            raise Exception("Unknown file format", save_file)


def convert_to_kmodel(tflite_path: str, options: dict, tool_options: dict = {"nncase_version": "1.9.0"}, verbose=False) -> str:
    """
    options should be of format {"target":"k210", "dataset":"path_to_dataset"}
    """
    if tool_options["nncase_version"] == "0.2":
        # perform manual
        return convert_to_kmodel_shell(tflite_path, options)
    else:
        return convert_to_kmodel_python(tflite_path, options=options, tool_options=tool_options, verbose=verbose)

def convert_to_kmodel_shell(tflite_path: str, options: dict) -> str:
    # if absent, download executables
    install_ncc_shell()
    # execute conversion
    kmodel_path = tflite_path.replace(".tflite", ".kmodel")
    os.system("{} compile {} {} -i tflite -o kmodel -t {} --dataset {}".format(
        os.path.join(NCC_PATH, "ncc"),
        tflite_path,
        kmodel_path,
        options["target"],
        options["dataset"]
    ))
    return kmodel_path


def convert_to_kmodel_python(tflite_path: str, options: dict = {"target": "k210"},
                             tool_options={"nncase_version": "1.9.0", "python_version": "3.9"}, verbose=False) -> str:
    try:
        import nncase
    except:
        install_ncc_python(version=tool_options["nncase_version"], python=tool_options["python_version"])
        import nncase

    kmodel_path = convert(tflite_path, options, verbose=verbose)
    return kmodel_path


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def convert(tflite_path, options, verbose=False):
    import nncase

    # model
    filename = tflite_path
    model = tflite_path

    if verbose: print ("\n\n=== Model description ===")
    if verbose: print("Detected format", filename.split('.')[-1])

    if "tflite" in filename:
        name = filename.split('.tflite')[0]
        # read TFLITE model and extract input shape
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        options["input_details"] = input_details
        options["output_details"] = output_details

        input_shapes = [inp['shape'] for inp in input_details]
        
        # check that the last dimension is channel
        if verbose: print("Transformation on the input shapes:")
        is_NHWC = False
        new_shapes = []
        for shape in input_shapes:
            is_NHWC = ((shape[-1] <= shape[-2]) & (shape[-1] <= shape[-3]))

            if is_NHWC and options.setdefault('layout', "NCHW") == 'NCHW':
                # if NHWC, change to NCHW, by swapping the last dimension and sending to index 1
                old_shape = shape.tolist()
                shape = [old_shape[0], old_shape[-1], old_shape[1], old_shape[2]]
                if verbose: print("\tNHWC -> NCHW : {} -> {}".format(old_shape, shape))
            new_shapes.append(shape)


        if len(input_shapes) == 1:
            input_shapes = new_shapes[0]

        options["input_shape"] = input_shapes

    elif "onnx" in filename:
        name = filename.split('.onnx')[0]
    else:
        raise Exception("Only TFLITE and ONNX are supported")
    target = 'k210'

    format = 'uint8'

    if verbose: print("\n=== Compilation phase ===")

    with suppress_output(verbose):
        # import
        model_content = read_model_file(model)
        # compile_options
        compile_options = nncase.CompileOptions()
        compile_options.target = options.setdefault("target", target)
        compile_options.quant_type = options.setdefault("quant_type", format)
        compile_options.input_range = options.setdefault("input_range", [0,255])
        compile_options.output_range = options.setdefault("output_range", [])
        compile_options.input_type = options.setdefault("input_type", format)
        # compile_options.output_type = options.setdefault("output_type", format)
        if len(options["input_shape"]) == 4: # case batch, C, H,W
            compile_options.input_shape = options["input_shape"]
            compile_options.input_layout = options.setdefault('layout', "NCHW")
            compile_options.output_layout = options.setdefault('layout', "NCHW")  # You can change this option to what you want.
        compile_options.model_layout = 'NHWC' if is_NHWC else 'NCHW'
        compile_options.preprocess = options.setdefault("preprocess", True)
        compile_options.mean = options.setdefault("mean", [0, 0, 0])
        compile_options.std = options.setdefault("std", [1, 1, 1])
        compile_options.dump_ir = True
        compile_options.dump_quant_error = True
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

        # post training quantization
        if options.setdefault("use_ptq", False):
            assert(options["calibration_dataset"] is not None)
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.samples_count = len(options["calibration_dataset"])
            calib_data = np.asarray(options["calibration_dataset"])
            ptq_options.set_tensor_data(calib_data.tobytes())
            compiler.use_ptq(ptq_options)

        # compile
        compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    if "onnx" in filename:
        name = name + "-onnx"
    kmodel_file = name + ".kmodel"
    with open(kmodel_file, 'wb') as f:
        f.write(kmodel)

    return kmodel_file

def simulate(kmodel_file, input_x, verbose=False, return_times = True):
    import nncase

    # simulate result
    start = time.time()
    kmodel = read_model_file(kmodel_file)
    simulator = nncase.Simulator()
    simulator.load_model(kmodel)
    model_load_time = float(time.time() - start)

    num_inputs = simulator.inputs_size
    num_outputs = simulator.outputs_size

    # check second axis of input_x is equal to num_inputs
    if num_inputs == 1:
        input_shape = simulator.get_input_tensor(0).shape
        if input_x[0].shape != input_shape:
            input_x = np.expand_dims(input_x, axis=1)
    else:
        if input_x.shape[1] != num_inputs:
            raise Exception("Input shape {} does not match number of inputs {}".format(input_x[0].shape, num_inputs))

    preds = []
    inference_time = []
    preprocess_time = []
    readout_times = []

    for i in range(len(input_x)):
        start = time.time()
        for tensor_i in range(num_inputs):
            input_tensor_i = simulator.get_input_tensor(tensor_i)
            tensor_type = str(input_tensor_i.dtype)
            dtype = numpy_types[tensor_type]

            input_xi = np.asarray(input_x[i:i+1][tensor_i], dtype=dtype)

            # check if input_xi is NHWC and convert to NCHW
            if input_xi.shape[-1] < input_xi.shape[-2] and input_xi.shape[-1] <= input_xi.shape[-3]:
                # NHWC -> NCHW
                input_xi = np.moveaxis(input_xi, -1, 1)

            # make sure input_xi matches tensor_i shape
            if input_xi.shape != input_tensor_i.shape:
                #warnings.warn("Input shape {} does not match tensor shape {}. Reshaping input...".format(input_xi.shape, input_tensor_i.shape))
                input_xi = input_xi.reshape(input_tensor_i.shape)
            rt_x = nncase.RuntimeTensor.from_numpy(input_xi)
            simulator.set_input_tensor(tensor_i, rt_x)
        preprocess_time.append(time.time() - start)
        # perform inference
        t = time.time()
        simulator.run()
        inference_time.append(time.time() - t)

        # read output
        outputs = []
        for tensor_o in range(num_outputs):
            start = time.time()
            output_tensor_o = simulator.get_output_tensor(tensor_o)
            readout_times.append(time.time() - start)
            pred = output_tensor_o.to_numpy()
            outputs.append(pred)
        while len(outputs)==1 and len(np.asarray(outputs).shape)>1:
            outputs = outputs[0] # remove extra dimension

        preds.append(outputs)

    if return_times:
        return preds, {"inference_ms": inference_time,
                       "load_ms": [model_load_time] + [0 for _ in range(len(inference_time)-1)],
                       "readout_ms": readout_times,
                       "preprocess_ms": preprocess_time}
    return preds
