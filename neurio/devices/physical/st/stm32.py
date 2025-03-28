#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 05.01.2024
Description: TODO
"""
import shutil
import warnings
import tensorflow as tf
from neurio.devices.device import Device
import os
import sys
import time

skeleton_projects = {
    "STM32L4R9": {"project": "STM32L4R9",
                  "project_name": "STM32L4R9",
                  "project_build": "STM32L4R9"},
    'NUCLEO-H723ZG': {"project": "NUCLEO-H723ZG",
                      "project_name": "NUCLEO-H723ZG",
                      "project_build": "NUCLEO-H723ZG"},
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")



class STM32(Device):
    def __init__(self, port: any = "serial", device_identifier: str = "STM32", name: str = "STM32Base",
                 log_dir: str = None, **kwargs):

        super().__init__(port, name, log_dir, **kwargs)
        self.verbose = kwargs.get("verbose", 0)
        self.device_identifier = device_identifier
        self.code_dir = os.path.join(self.log_dir, "c_project")
        os.makedirs(self.code_dir, exist_ok=True)
        # check for env variables
        self.__check__env__()

    def __check__env__(self):
        # check STM32CubeProgrammer is downlaoded
        if os.environ["STM32CUBEPROGRAMER_CLI_PATH"] is None:
            raise Exception(
                "STM32CubeProgramer path not found, please set the STM32CUBEPROGRAMER_PATH environment variable to the path of STM32CubeProgramer")
        if os.environ["STM32CUBEIDE_PATH"] is None:
            raise Exception(
                "STM32CubeIDE path not found, please set the STM32CUBEIDE_PATH environment variable to the path of STM32CubeIDE")
        if os.environ["X_CUBE_AI_PATH"] is None:
            raise Exception(
                "X-CUBE-AI path not found, please set the X_CUBE_AI_PATH environment variable to the path of X-CUBE-AI")
        self.programmer_cli_path = os.environ["STM32CUBEPROGRAMER_CLI_PATH"]
        self.x_cube_ai_path = os.environ["X_CUBE_AI_PATH"]
        self.stm32ai_exec = os.path.join(self.x_cube_ai_path, "Utilities", "mac", "stm32ai")
        self.stm32ai_running_lib_path = os.path.join(self.x_cube_ai_path, "scripts", "ai_runner")
        self.cube_ide_path = os.environ["STM32CUBEIDE_PATH"]
        sys.path.append(self.stm32ai_running_lib_path)

    def __prepare_model__(self, model, **kwargs):
        self.original_model = model
        if "quantize" in kwargs:
            self.quantize = kwargs["quantize"]
        else:
            self.quantize = False

        #self.model_path = os.path.join(self.log_dir, "original_model.h5")
        #self.original_model.save(self.model_path)
        # convert to tflite
        self.tflite_model_path = os.path.join(self.log_dir, "tflite_model.tflite")
        if isinstance(model, str) and model.endswith(".tflite"):
            # no need for conversion
            shutil.copy(model, self.tflite_model_path)
            print("Model is already in tflite format. Copying it to log directory")
            self.model_path = self.tflite_model_path

        elif isinstance(model, str) and model.endswith(".h5"):
            # load model from keras
            model = tf.keras.models.load_model(model)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(self.tflite_model_path, "wb") as f:
                f.write(tflite_model)
            self.model_path = self.tflite_model_path

        elif isinstance(model, str) and model.endswith(".onnx"):
            self.tflite_model_path = os.path.join(self.log_dir, "onnx_model.onnx")
            shutil.copy(model, self.tflite_model_path)
            print("Model is in onnx format. Copying it to log directory")
            self.model_path = self.tflite_model_path

        elif isinstance(model, tf.keras.Model):
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(self.tflite_model_path, "wb") as f:
                f.write(tflite_model)
            self.model_path = self.tflite_model_path
        else:
            raise Exception("Unknown model format")


    def __generate_network_code__(self):
        # generate code for the network
        if self.verbose > 0:
            print("Generating model c files: {}".format(self.model_path))
        compression = "high" if self.quantize else None
        command = " ".join(
            ["{} generate".format(self.stm32ai_exec), '-m', '{}'.format(self.model_path), '-o',
             '{}/{}'.format(ASSETS_DIR, skeleton_projects[self.device_identifier]["project"])])

        if compression is not None:
            command += " ".join([' -c', '{}'.format(compression)])
        if self.verbose > 0:
            print(command)

        os.system(command)

    def __compile_project__(self):
        # clean all
        print("Building elf...")
        command = " ".join([self.cube_ide_path,
                            "-nosplash -application org.eclipse.cdt.managedbuilder.core.headlessbuild",
                            '-data', '{}'.format(ASSETS_DIR),
                            '-importAll',
                            '{}/{}'.format(ASSETS_DIR, skeleton_projects[self.device_identifier]["project"]),
                            '-cleanBuild', '{}'.format(skeleton_projects[self.device_identifier]["project_name"])])
        print(command)
        os.system(command)

    def __generate_inference_code__(self):
        # code is already generated in assets. We need to select the right one
        # based on the device
        self.project = skeleton_projects[self.device_identifier]["project"]
        self.project_name = skeleton_projects[self.device_identifier]["project_name"]
        self.project_build = skeleton_projects[self.device_identifier]["project_build"]
        self.project_path = os.path.join(ASSETS_DIR, self.project)

        # generate code for the network
        self.__generate_network_code__()

        # compile the entire project
        self.__compile_project__()

    def __deploy_model__(self):
        print("Model is already deployed within the code")
        pass

    def __deploy_code__(self):
        # deploy the code
        if self.verbose > 0: print("Flashing...")
        command = " ".join([self.programmer_cli_path, '-c', 'port=SWD', '-w',
                            '{}/{}/Debug/{}.elf'.format(ASSETS_DIR,
                                                        skeleton_projects[self.device_identifier]["project_build"],
                                                        skeleton_projects[self.device_identifier]["project_name"])]
                           )
        if self.verbose > 0: print(command)
        os.system(command)

        time.sleep(2)
        self.__reset_device__()

    def __reset_device__(self):
        command = " ".join([self.programmer_cli_path, '-c', 'port=SWD', 'mode=POWERDOWN', '-s', '0x08000000'])
        os.system(command)

        time.sleep(2)
        command = " ".join([self.programmer_cli_path, '-c', 'port=SWD', '-rst', '-s', '-hardRst'])
        os.system(command)
        if self.verbose > 0: print("Device reset")
        time.sleep(2)
        self.__connect_runner()

    def __prepare_data__(self, input_x, **kwargs):
        # default: convert to float32
        input_x_processed = input_x.astype("float32")
        return input_x_processed

    def __connect_runner(self):
        try:
            from stm_ai_runner import AiRunner  # imported from X-CUBE-AI path
        except ImportError:
            raise Exception(
                "stm_ai_runner not found, please set the X_CUBE_AI_PATH environment variable to the path of X-CUBE-AI")

        runner = AiRunner(debug=True)
        # Connect to device
        if not runner.connect(self.port):
            raise RuntimeError('Error when connecting "{}": {}'.format(self.port, runner._last_err))
        print(runner, flush=True)
        runner.summary()
        self.runner = runner

    def __transfer_data_to_memory__(self, input_x):
        # Initialize Ai Runner
        warnings.warn("Data is transfered during inference, no need to transfer it before")
        self.inputs = input_x

    def __run_inference__(self, profile: bool = True):
        if self.runner is None:
            raise Exception("Runner is not initialized")

        # load the dataset
        inputs = self.inputs
        try:
            predictions, profiler = self.runner.invoke(inputs)
        except:
            print("[NeurIO ERROR]: {}".format(self.runner._last_err))
            # runner not connected, reconnect
            self.__connect_runner()
            time.sleep(2)
            predictions, profiler = self.runner.invoke(inputs)


        self.tmp_predictions = predictions
        self.tmp_profiler = profiler


    def __read_inference_results__(self):
        warnings.warn("Data is send back by AiRunner after inference, no need to transfer it after")
        return self.tmp_predictions, self.tmp_profiler

    def is_alive(self, timeout: int = 20) -> bool:
        pass

    def __str__(self):
        pass


class STM32L4R9(STM32):
    def __init__(self, port: any = "serial", name: str = "STM32L4R9I-DISCO", log_dir: str = None, **kwargs):
        super().__init__(port=port, device_identifier="STM32L4R9", name=name, log_dir=log_dir)


class NUCLEOH723ZG(STM32):
    def __init__(self, port: any = "serial", name: str = "NUCLEO-H723ZG",
                 log_dir: str = None, **kwargs):
        super().__init__(port=port, device_identifier="NUCLEO-H723ZG", name=name, log_dir=log_dir)