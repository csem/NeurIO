#!/user/bin/env python

from neurio.tasks.task import Task
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score

_DOWNLOAD_PATH = (
    'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
)
_TEST_DOWNLOAD_PATH_ = (
    'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
)

_SPLITS = ['train', 'valid', 'test']

WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = '_background_noise_'
SAMPLE_RATE = 16000


class SpeechCommands(Task):
    """
    Keyword Spotting task based on the SpeechCommand dataset, from the paper:
    Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition."

    The task consists in classifying a 1-second audio clip into one of the following classes:

    - "yes"
    - "no"
    - "up"
    - "down"
    - "left"
    - "right"
    - "on"
    - "off"
    - "stop"
    - "go"
    - "unknown"
    - "silence"

    The "unknown" class is used for words that are not in the list above, and the "silence" class is used for silence.

    The dataset is composed of 105829 training samples, 158538 testing samples and 11005 validation samples.

    The task is evaluated using the accuracy metric.

    The dataset can be downloaded from https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    """

    def __init__(self):
        super(SpeechCommands, self).__init__()
        data_save_dir = os.path.join(os.path.dirname(__file__), "data/speech_commands")
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        # Download the dataset if it does not exist
        self.data_path = os.path.join(os.path.dirname(__file__), "data/speech_commands")
        self.__download_and_extract()
        self.__read_data__()

        self.metrics = {"accuracy": accuracy_score,
                        "balanced_accuracy": balanced_accuracy_score}

    def __download_and_extract(self):
        dataset_archive_path = os.path.join(self.data_path, "speech_commands_v0.02.tar.gz")
        if not os.path.exists(dataset_archive_path):
            print("Downloading SpeechCommands v0.02 dataset...")
            # download data from url and save it locally, crossplatform compatible

            # import urllib.request
            url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

            # urllib.request.urlretrieve(url, dataset_archive_path)

            # import wget
            # wget.download(url, dataset_archive_path)

            import requests
            r = requests.get(url, allow_redirects=True)
            open(dataset_archive_path, 'wb').write(r.content)

        # Extract the dataset if it does not exist
        if not os.path.exists(os.path.join(self.data_path, "LICENSE")):
            import tarfile
            tar = tarfile.open(dataset_archive_path, "r:gz")
            tar.extractall(self.data_path)
            tar.close()

    def __read_data__(self):
        """
        Reads the data from the dataset and returns it as a list of tuples (x, y), and assign the values to the corresponding
        partition.
        """

        test_file = os.path.join(self.data_path, "testing_list.txt")
        with open(test_file, 'r') as f:
            test_files = f.readlines()

        valid_file = os.path.join(self.data_path, "validation_list.txt")
        with open(valid_file, 'r') as f:
            validation_files = f.readlines()

        test_files = [os.path.join(self.data_path, file.strip()) for file in test_files if
                      file.strip().split("/")[0] in WORDS]
        validation_files = [os.path.join(self.data_path, file.strip()) for file in validation_files if
                            file.strip().split("/")[0] in WORDS]

        # list all files from data
        data_files = []
        for word in WORDS:
            word_path = os.path.join(self.data_path, word)
            data_files += [os.path.join(word_path, file) for file in os.listdir(word_path)]

        # remove test and validation files from data files
        data_files = [file for file in data_files if file not in test_files and file not in validation_files]

        # create the dataset by reading the files
        label_dict = {
            "_silence_": 0,
            "_unknown_": 1,
            "down": 2,
            "go": 3,
            "left": 4,
            "no": 5,
            "off": 6,
            "on": 7,
            "right": 8,
            "stop": 9,
            "up": 10,
            "yes": 11,
        }

        # read the wav files and create the dataset
        import librosa
        import numpy as np
        import tensorflow as tf

        def read_wav_file(file):
            audio, sample_rate = librosa.load(file, sr=SAMPLE_RATE)
            return audio, sample_rate

        # generate numpy dataset from wav files
        def generate_dataset(files, threads=4):
            # parallel processing

            def _parse_function(file):
                audio, sample_rate = read_wav_file(file)
                label = file.split("/")[-2]
                return audio, label

            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.map(
                lambda file: tuple(tf.py_function(_parse_function, [file], [tf.int16, tf.string])),
                num_parallel_calls=threads)
            dataset = dataset.map(
                lambda audio, label: (audio, tf.py_function(lambda label: label_dict[label], [label], tf.int16)))

            # convert to numpy
            dataset = dataset.map(
                lambda audio, label: (tf.py_function(lambda audio: np.array(audio), [audio], tf.int16), label))
            return dataset

        # generate the dataset
        train_dataset = generate_dataset(data_files)
        test_dataset = generate_dataset(test_files)
        validation_dataset = generate_dataset(validation_files)

        self.train_data = train_dataset
        self.test_data = test_dataset
        self.validation_data = validation_dataset

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data

    def get_metrics_info(self) -> dict:
        pass

    def get_metrics(self):
        return self.metrics

    def __str__(self) -> str:
        pass
