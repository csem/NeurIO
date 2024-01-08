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

from neurio.devices.device import Device
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
from neurio.converters.kendryte_utils import convert_to_kmodel, simulate
from neurio.converters.tflite_utils import keras_to_tflite
from neurio.helpers import NpEncoder
import tensorflow as tf
from neurio.exceptions import InvalidImageRangeError
from neurio.benchmarking.profiler import Profiler

NNCASE_VERSION = "1.9.0"
python_version = sys.version_info
PYTHON_VERSION = "{}.{}.{}".format(python_version.major, python_version.minor, python_version.micro)


class K210Virtual(Device):

    def __init__(self, port: any = "virtual", name: str = "K210", log_dir: str = None, **kwargs):
        super().__init__(port, name, log_dir, **kwargs)
        if self.port != "virtual":
            raise Exception("K210VirtualDevice only supports virtual ports")
        self.tools_options = {"nncase_version": NNCASE_VERSION,
                              "python_version": PYTHON_VERSION}

        # self.image_format = kwargs.get("image_format", "jpeg") # works well on deployment
        self.image_format = kwargs.get("image_format", "bmp")
        self.verbose = kwargs.get("verbose", 0)

        # generate logdir based on current date and time if None
        if self.log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.output_dir = os.path.join(self.log_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.profiler = Profiler()

    def is_alive(self, timeout: int = 20) -> bool:
        return True

    def __prepare_model__(self, model: tf.keras.models.Model, **kwargs):
        """
        Prepare the model for deployment. This includes converting it to a tflite model and then to a kmodel.

        :param model: Model to be prepared
        :param kwargs: Other parameters
        """

        self.profiler["model"]["model_name"] = model.name
        self.profiler["model"]["framework"] = "tensorflow"
        self.profiler["model"]["model_datetime"] = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

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
        self.profiler["model"]["compile_datetime"] = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        self.profiler["model"]["parameters"] = model.count_params()

        self.kmodel_path = os.path.abspath(kmodel_file)

        # get input and output shapes of the model and store in profiler
        self.input_shapes = model.input_shape
        self.output_shapes = model.output_shape

        self.profiler["model"]["inputs"] = [{"name": i.name, "shape": i.shape} for i in model.inputs]
        self.profiler["model"]["outputs"] = [{"name": i.name, "shape": i.shape} for i in model.outputs]
        # TODO
        #self.profiler["model"]["macc"] = 2*model.count_params()

        # complete profiler with device details
        self.__populate_profiler__()


    def __generate_inference_code__(self):
        pass

    def __deploy_model__(self):
        pass

    def __deploy_code__(self):
        pass

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

    def __subsample_data__(self, input_x, batch_size: int, batch_index: int):
        """
                Subsample a batch of data from the dataset.

                :param data: the dataset
                :param batch_size: the batch size
                :param batch_index: the index of the batch
                :return: the batch of data
                """
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        return input_x[start_index:end_index]

    def __transfer_data_to_memory__(self, input_x):
        """
        Transfer the data to the RAM, by setting the value of self.input_data
        :param input_x: the input data
        """
        self.input_data = input_x

    def __run_inference__(self, profile: bool = True):
        load_times = []
        preprocess_times = []
        inference_times = []
        readout_times = []
        y_pred = []

        # load data
        data = self.input_data
        imgs = data
        imgs = [np.array(x) for x in imgs]
        imgs = [np.expand_dims(x, axis=0) for x in imgs]
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs = np.asarray(imgs, dtype=np.uint8)

        outputs, times = simulate(self.kmodel_path, imgs, return_times=True)

        y_pred += outputs
        inference_times += times["inference_ms"]
        preprocess_times += times["preprocess_ms"]
        load_times += times["load_ms"]
        readout_times += times["readout_ms"]

        results = {
            "output": y_pred,
            "inference_ms": inference_times,
            "preprocess_ms": preprocess_times,
            "load_model_ms": load_times,
            "readout_ms": readout_times,
        }
        # save results in json
        json_filename = os.path.join(self.output_dir, "results.json")
        with open(json_filename, "w") as f:
            json.dump(results, f, cls=NpEncoder)

    def __read_inference_results__(self):
        json_filename = os.path.join(self.output_dir, "results.json")
        with open(json_filename, "r") as f:
            results = json.load(f)

        temp_profiler = Profiler(self.profiler.json_data) # get copy of current profiler
        temp_profiler["inference"]["batch_size"] = len(self.input_data)
        temp_profiler["inference"]["inference_time"] = results["inference_ms"]
        temp_profiler["inference"]["preprocess_time"] = results["preprocess_ms"]
        temp_profiler["inference"]["model_load_time"] = results["load_model_ms"]
        temp_profiler["inference"]["readout_time"] = results["readout_ms"]

        return results["output"], temp_profiler

    def __populate_profiler__(self):
        # setting profiler info
        self.profiler["device"]["name"] = "K210 Emulator"
        self.profiler["device"]["port"] = "virtual"
        self.profiler["device"]["dev_type"] = "k210"
        self.profiler["device"]["system"] = "maixpy 0.6.2"
        self.profiler["device"]["cpu_clock"] = 400000000
        self.profiler["device"]["kpu_clock"] = 400000000
        self.profiler["device"]["attrs"] = [
            "fpu",
            "kpu",
            # TODO complete
        ]
        self.profiler["runtime"]["name"] = "Custom MaixPy 0.1"
        self.profiler["runtime"]["version"] = [0, 1]
        self.profiler["runtime"]["tools_version"] = {
            "tensorflow": [int(x) if x.isnumeric() else x for x in tf.__version__.split(".")],
            "maixpy": [0, 6, 2],  # TODO automatically get from board
            "ncc": [int(x) if x.isnumeric() else x for x in NNCASE_VERSION.split(".")],
            "kmodel": "v5" if NNCASE_VERSION.split(".")[0] == "1" else "v4",
            "tflite": [int(x) if x.isnumeric() else x for x in tf.__version__.split(".")],
        }


    def __str__(self):
        return str({
            "port": self.port,
            "name": self.name,
            "log_dir": self.log_dir,
        })
