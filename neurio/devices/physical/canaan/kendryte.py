#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 21.09.22
Description: K210 deployment pipeline
"""
import datetime
from typing import Tuple, Optional

import sys

sys.path.append("../../")
sys.path.append("../../../../")
from neurio.common import Prediction, Profiler
from neurio.devices.device import Device
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
from neurio.converters.kendryte_utils import convert_to_kmodel
from neurio.converters.tflite_utils import keras_to_tflite
from neurio.exceptions import InvalidImageRangeError
import tensorflow as tf
import time
import pexpect

NNCASE_VERSION = "1.9.0"
python_version = sys.version_info
PYTHON_VERSION = "{}.{}.{}".format(python_version.major, python_version.minor, python_version.micro)


class K210(Device):

    def __init__(self, port: any, name: str = "k210", log_dir: str = None, **kwargs):
        super().__init__(port, name, log_dir, **kwargs)
        self.original_model = None
        self.kmodel = None
        self.tflite_model = None
        self.verbose = kwargs.get("verbose", 0)
        self.device_storage_location = kwargs.get("device_storage_location", "/sd")
        self.rshell_session = None
        self.rshell_session = self.__create_rshell_session__(port)
        self.code_dir = os.path.join(self.log_dir, "code")
        os.makedirs(self.code_dir, exist_ok=True)

    def __prepare_model__(self, model: tf.keras.models.Model, **kwargs):
        """
        Prepare the model for deployment. This includes converting it to a tflite model and then to a kmodel.

        :param model: Model to be prepared
        :param kwargs: Other parameters
        """
        self.models_dir = os.path.join(self.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.calibration_dir = os.path.join(self.log_dir, "calibration")
        os.makedirs(self.calibration_dir, exist_ok=True)
        # save calibration directory passed in kwargs
        if "calibration_data" in kwargs:
            for data in kwargs["calibration_data"]:
                img, label = data
                img.save(os.path.join(self.calibration_dir, "{}.bmp".format(label)))

        tflite_model = keras_to_tflite(model, tflite_path=os.path.join(self.models_dir, model.name + ".tflite"))
        compilation_options = {"target": "k210",
                               "dataset": os.path.abspath(self.calibration_dir)}

        kmodel_file = convert_to_kmodel(tflite_model,
                                        options=compilation_options,
                                        tool_options={"nncase_version": NNCASE_VERSION,
                                                      "python_version": PYTHON_VERSION},
                                        verbose=self.verbose)
        self.kmodel_path = os.path.abspath(kmodel_file)

        # get input and output shapes of the model
        self.input_shapes = model.input_shape
        self.output_shapes = model.output_shape

    def __generate_inference_code__(self):
        """
        Generate the inference code for the device.
        The script will be generated in MicroPython and stored in the "upload" directory of the logs.
        """
        # edit benchmark.py to fit data and model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        skeleton_dir = os.path.join(dir_path, "assets", "k210", "skeletons", "v{}".format(NNCASE_VERSION))
        with open(os.path.join(skeleton_dir, "benchmark.py"), "r") as f:
            file = f.read()

        file = file.replace("DEBUG = True", "DEBUG = {}".format(str(self.verbose).title()))
        file = file.replace("IMG_WIDTH = 224", "IMG_WIDTH = {}".format(self.input_shapes[0]))
        file = file.replace("IMG_HEIGHT = 224", "IMG_HEIGHT = {}".format(self.input_shapes[1]))
        file = file.replace("KPU_OUTPUT = [0, 1, 1, 5]", "KPU_OUTPUT = {}".format(self.output_shapes))

        # sfile = file.replace('SD_FOLDER_DATA = "/sd/data"', 'SD_FOLDER_DATA = "{}"'.format("/sd/img"))
        file = file.replace('SD_RESULT_FILE = "/sd/results.txt"',
                            'SD_RESULT_FILE = "{}"'.format("{}/results.txt".format(self.device_storage_location)))
        file = file.replace('MODEL_PATH = "/sd/test.kmodel"',
                            'MODEL_PATH = "{}"'.format(
                                "{}/{}".format(self.device_storage_location, os.path.basename(self.kmodel_path))))

        # copy skeletons of prepare.py and benchmark.py
        with open(os.path.join(self.code_dir, "benchmark.py"), "w") as f:
            f.write(file)

        with open(os.path.join(skeleton_dir, "prepare.py"), "r") as p:
            prepare_content = p.read()
            with open(os.path.join(self.code_dir, "prepare.py"), "w") as f:
                f.write(prepare_content)

    def __deploy_model__(self):
        """
        Deploy the model on the device by copying it to the device storage location (usually /sd).
        The model is uploaded via rshell (UART connection), and the upload time is estimated based on the size of the model and the upload speed.
        This can take a while, depending on the size of the model.
        """
        if self.verbose > 0: print("Size of model: {} MB".format(os.path.getsize(self.kmodel_path) / 1024 / 1024))
        upload_speed = 14  # KB/s
        if self.verbose > 0: print("Upload speed: {} KB/s".format(upload_speed))
        estimated_upload_time = os.path.getsize(self.kmodel_path) / 1024 / upload_speed
        if self.verbose > 0: print("Estimated upload time: {} s".format(estimated_upload_time))
        upload_start_time = time.time()
        self.__execute_command_sync__("cp {} {}/".format(self.kmodel_path, self.device_storage_location),
                                      timeout=int(10 * estimated_upload_time))
        if self.verbose > 0: print("Upload done in {} s".format(time.time() - upload_start_time))

    def __deploy_code__(self):
        """
        Deploy the code for inference on the device by copying it to the device storage location (usually /sd).
        The code consists in two MicroPython scripts: prepare.py and benchmark.py
        The prepare.py script is used to set the device up for inference, while the benchmark.py script is used to run the inference.
        Both are uploaded via rshell (UART connection).
        """
        self.__execute_command_sync__(
            "cp {} {}/".format(os.path.join(self.code_dir, "benchmark.py"),
                                self.device_storage_location), 120)
        self.__execute_command_sync__(
            "cp {} {}/".format(os.path.join(self.code_dir, "prepare.py"), self.device_storage_location),
            120)

    def __prepare_data__(self, input_x, **kwargs):

        def __check_image_range__(image):
            """
            Checks if the range of an image corresponds to its format.

            :param: image_path (str): Path to the input image file.
            :return:
                bool: True if the range is correct, False otherwise.
            """
            # Get the format of the image
            image_format = image.format

            # Define the expected range based on the image format
            expected_range = {
                "JPEG": (0, 255),  # JPEG format uses 8-bit per channel (0-255)
                "PNG": (0, 255),  # PNG format also uses 8-bit per channel (0-255)
                "BMP": (0, 255),
                None: (0, 255),  # None format typically uses 8-bit per channel (0-255)
                "PPM": (0, 255),  # PPM format also uses 8-bit per channel (0-255),
            }

            # Check if the pixel values are within the expected range
            min_value, max_value = expected_range.get(image_format, (0, 0))
            p_min = min(image.getdata())
            p_max = max(image.getdata())

            # case 1-channel image
            if isinstance(p_min, int) and p_min >= min_value and p_max <= max_value:
                return True
            # case 3-channel image
            elif p_min[0] >= min_value and p_max[0] <= max_value and p_min[1] >= min_value and p_max[1] <= max_value and \
                    p_min[2] >= min_value and p_max[2] <= max_value:
                return True
            else:
                return False

        def __preprocess_image__(img):
            """
            Preprocess an image by converting it to the appropriate format and range.

            :param img: image to preprocess
            :return: the preprocessed image
            """
            # check if im has dimension channel dimension of 1. If yes, remove dimension.
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = np.squeeze(img, axis=2)

            # Create a PIL image from the input NumPy array
            image = Image.fromarray(img)

            # Check if the image range corresponds to its format
            if not __check_image_range__(image):
                raise InvalidImageRangeError("The image range does not correspond to its format.")

            # Get the number of channels in the image, and determine the appropriate mode based on the number of channels
            num_channels = img.shape[2] if len(img.shape) == 3 else 1
            if num_channels == 1:  # Grayscale image
                mode = "L"
            elif num_channels == 3:  # RGB image
                mode = "RGB"
            elif num_channels == 4:  # RGBA image
                mode = "RGBA"
            else:
                if self.verbose > 0: print("Unsupported number of channels: {}".format(num_channels))
                return False

            # Convert the image to the determined mode
            image = image.convert(mode)

            # Save the image as BMP
            return image

        if self.verbose > 0: print("Saving dataset")

        # preprocess data
        preprocessed_data = []
        for i in tqdm(range(len(input_x))):
            data_x = input_x[i]
            img = __preprocess_image__(data_x)
            preprocessed_data.append(img)

        return preprocessed_data

    def __transfer_data_to_memory__(self, input_x):
        """
        Transfer the data to the device memory.

        :param input_x: the data to transfer
        """
        # clean data folder
        self.__execute_command_sync__('rm -rf {}/data/*\n'.format(self.device_storage_location), timeout=20)
        # create data folder
        self.__execute_command_sync__('mkdir {}/data\n'.format(self.device_storage_location), timeout=20)
        # clean results folder
        self.__execute_command_sync__('rm -rf {}/results.txt\n'.format(self.device_storage_location), timeout=20)

        # create folders to store images
        self.data_dir = os.path.join(self.log_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        json_data = {}
        for i in range(len(input_x)):
            img = input_x[i]
            if self.verbose > 0: print("Uploading data to SD card")
            # save image as BMP
            img_filename = os.path.join(self.data_dir, "{}.bmp".format(i))
            img.save(img_filename)
            # upload the saved sample to the "data" folder in the memory of the device
            destination = "{}/data/{}".format(self.device_storage_location, os.path.basename(img_filename))
            self.__execute_command_sync__('cp {} {}\n'.format(img_filename, destination), timeout=20)

            # generate json file with filenames
            json_data[i] = os.path.basename(img_filename)

        # upload json too
        json_filename = os.path.join(self.log_dir, "data/img_list.json")
        with open(json_filename, "w") as f:
            json.dump(json_data, f)
        device_json_filename = "{}/inference.json".format(self.device_storage_location)
        self.__execute_command_sync__('cp {} {}\n'.format(json_filename, device_json_filename), timeout=20)

    def __run_inference__(self, profile: bool = True):
        """
        Triggers the inference on the device.

        :param profile: Has no effect in K210, profiling always occur.
        """

        # Open Micropython REPL
        if self.verbose > 0: print("[RSHELL] {}".format("repl"))
        self.rshell_session.sendline("repl")
        self.rshell_session.expect("MicroPython", timeout=15)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        time.sleep(0.5)
        self.rshell_session.expect(">>>", timeout=15)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())

        # import scripts
        if self.verbose > 0: print("[RSHELL] {}".format("import benchmark"))
        self.rshell_session.sendline("import benchmark")
        self.rshell_session.expect(">", timeout=15)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        time.sleep(0.5)

        # run inference
        if self.verbose > 0: print("[RSHELL] {}".format("benchmark.main()"))
        self.rshell_session.sendline("benchmark.main()")
        index = self.rshell_session.expect(["Done", "memory", "MemoryError"], timeout=20 * 60)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        # check if a problem occurred
        if index == 0:
            # if no problem
            if self.verbose > 0: print("Done")
            self.rshell_session.expect("MicroPython", timeout=20)  # wait for restart
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())
            self.rshell_session.expect("information.", timeout=20)
            time.sleep(0.5)
            # exit REPL
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())
            self.rshell_session.sendcontrol("X")  # ctrl+x
            expected_patterns = ['/[^>]*>', '/sd>[^>]*', '/flash>[^>]*']
            pattern_index = self.rshell_session.expect(expected_patterns, timeout=20)
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())
            time.sleep(0.1)

            # wait for reset
            if pattern_index == 0 or pattern_index == 1 or pattern_index == 2:
                if self.verbose > 0:  print("Machine reset.")
            time.sleep(0.1)
            self.rshell_session.sendline("ls {}".format(self.device_storage_location))
            self.rshell_session.expect(["benchmark.py", pexpect.EOF], timeout=120)
            if self.verbose > 0:
                print(self.rshell_session.before.decode().strip())
                self.rshell_session.before = ""
                print("\033[0m")
            time.sleep(1.0)

        # Memory error due to KPU
        if index == 1:
            self.__reset_device__()
            if self.verbose > 0: print("KPU Memory error: could not load kmodel in KPU.")
            raise MemoryError("Out of memory")

        # Memory error in MicroPython
        if index == 2:
            self.__reset_device__()
            if self.verbose > 0: print("Memory error during inference step on MicroPython.")
            raise MemoryError("Memory error during inference")

    # Helper methods
    def __read_inference_results__(self):
        """
        Read results from devices, by downloading the results file from the device and parsing it.

        :return: tuple of (predictions, profiler)
        """
        def text_to_json(text):
            data = text

            data = "[" + data
            data = data.replace("(", "'(")
            data = data.replace(")", ")'")
            data = data.replace("}{", "},{")
            data = data + "]"

            data = data.replace("'", '"')
            return data

        def _read_result_file(filepath: str) -> any:
            with open(filepath, "r") as f:
                if filepath.endswith(".txt"):
                    text_data = f.read()
                    json_data = text_to_json(text_data)
                    json_data = json_data.replace("//./////", "\"nan\"")
                    data = json.loads(json_data)
                elif filepath.endswith(".json"):
                    data = json.load(f)
                else:
                    raise Exception("File extension not known")
            return data

        inference_times = []
        preprocess_times = []
        load_times = []
        y_pred = []

        self.__download_results__()
        result_file = os.path.join(self.results_dir, "results.txt")
        predicted_json = _read_result_file(result_file)

        # sort by input index
        predicted_json = sorted(predicted_json, key=lambda x: int(x['filename'].split('/')[-1].split('.')[0]))

        for r_dict in predicted_json:
            filename = os.path.basename(r_dict["filename"])
            inference = r_dict["inference_ms"]
            load = r_dict["load_ms"]
            preprocess = r_dict["preprocess_ms"]
            if "nan" not in r_dict["output"]:
                y_p = np.asarray(r_dict["output"])
            else:
                y_p = r_dict["output"]
            # store
            inference_times.append(inference)
            load_times.append(load)
            preprocess_times.append(preprocess)
            y_pred.append(y_p)

        # predictions and Profiler
        temp_profiler = Profiler()
        temp_profiler.inference_times = inference_times
        temp_profiler.load_times = load_times
        temp_profiler.preprocess_times = preprocess_times

        return y_pred, temp_profiler

    def __download_results__(self):
        """
        Download the results of the inference from the device to the local storage.
        """

        self.results_dir = os.path.join(self.log_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        cmd = "ls /flash"
        if self.verbose > 0: print("[RSHELL] {}".format(cmd))
        self.rshell_session.sendline(cmd)
        self.rshell_session.expect("freq.conf", timeout=120)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        cmd = "cp {}/results.txt {}/".format(self.device_storage_location, self.results_dir)
        if self.verbose > 0: print("[RSHELL] {}".format(cmd))
        self.rshell_session.sendline(cmd)
        self.rshell_session.expect("Copying", timeout=10)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        self.rshell_session.expect(">", timeout=10)
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())

    def __reset_device__(self):
        """
        Reset the device by sending a reset command to the device.
        """
        if self.verbose > 0: print("====>>>>>  RESETTING DEVICE")
        if self.verbose > 0: print(self.rshell_session.before.decode().strip())

        # check if in repl mode
        in_repl = True
        try:
            cmd = "ls"
            if self.verbose > 0: print("[RSHELL] {}".format(cmd))
            self.rshell_session.sendline(cmd)
            self.rshell_session.expect("NameError", timeout=2)
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())
        except pexpect.exceptions.TIMEOUT:
            in_repl = False

        if in_repl:
            # send reset and quit repl
            cmd = "import machine"
            if self.verbose > 0: print("[RSHELL] {}".format(cmd))
            self.rshell_session.sendline(cmd)
            self.rshell_session.expect(">", timeout=10)
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())

            cmd = "machine.reset()"
            if self.verbose > 0: print("[RSHELL] {}".format(cmd))
            self.rshell_session.sendline(cmd)
            self.rshell_session.expect(">", timeout=10)
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())

            time.sleep(1.0)  # ctrl+x
            self.rshell_session.sendcontrol("X")  # ctrl+x
            self.rshell_session.expect(">", timeout=10)
            if self.verbose > 0: print(self.rshell_session.before.decode().strip())
            time.sleep(1.0)

            if self.verbose > 0: print("Device reset.")

    def __create_rshell_session__(self, port):
        """
        Create a rshell session to the device.

        :param port: the port to which the device is connected
        :return: a pexpect session from the rshell package, that can be used as pipe to send commands to the device
        """
        command = "rshell -p {}".format(port)
        child = pexpect.spawn(command)
        try:
            child.expect("connected", timeout=20)
        except Exception as e:
            raise ConnectionError("Device not available at {}".format(port))

        if self.verbose > 0: print(child.before.decode().strip())
        if self.verbose > 0: print("Connecting...")
        child.expect("Retrieving sysname ...", timeout=10)
        if self.verbose > 0: print(child.before.decode().strip())
        child.expect("CanMV", timeout=10)
        if self.verbose > 0: print("Connected. ")
        child.expect(">", timeout=10)
        output = child.before.decode().strip()
        if self.verbose > 0: print("Output:", output)
        return child

    def __execute_command_sync__(self, cmd: str, timeout=20):
        """
        Execute a command on the device and wait for the result.

        :param cmd: the command line to send through rshell
        :param timeout: the timeout for the command, in seconds
        :return: the output of the command
        """
        stripped_command = cmd.replace("rshell -p {}".format(self.port), "")
        stripped_command = stripped_command.replace("-f", "").strip()
        if len(stripped_command) > 1:
            if self.verbose > 0: print("[RSHELL] {}".format(stripped_command))
            self.rshell_session.sendline(stripped_command)
            self.rshell_session.expect(">", timeout=timeout)
            out = self.rshell_session.before.decode().strip()
            if self.verbose > 0: print(out)
            return out
        else:
            if self.verbose > 0: print("Skipping empty command")
            return ""

    def is_alive(self, timeout: int = 20) -> bool:
        pass

    def __str__(self):
        pass
