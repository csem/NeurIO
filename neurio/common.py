import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


@dataclass
class Prediction:
    pred: np.ndarray
    prof: any

    def __iter__(self):
        yield from dataclasses.asdict(self).values()


@dataclass
class Benchmarking:
    res: any

    def __iter__(self):
        yield from dataclasses.asdict(self).values()


@dataclass
class Profiler:
    """
    Profiler class

    """
    json_file: Any
    json_data: Dict[str, Any] = None

    def load(self) -> Dict[str, Any]:
        """
        Load json data from file
        :return: Dict[str, Any] containing the json data
        """
        with open(self.json_file, 'r*') as file:
            self.json_data = json.load(file)
        return self.json_data

    def save(self, filepath: str):
        """
        Save json data to file
        :param outfile: where to save the data of the profiler, as json file
        """
        if not filepath.endswith('.json'):
            raise ValueError('The path must be a json file')
        with open(filepath, 'w') as file:
            json.dump(self.json_data, file)

    def __post_init__(self):
        """
        Post init function that load the json data from the file and set the attributes.
        """
        if self.json_data is None:
            self.json_data = self.load_json_data()
        for key, value in self.json_data.items():
            setattr(self, key, value)
