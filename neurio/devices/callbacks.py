#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 12.01.2024
Description: Classes that can be used as power monitors
"""
from abc import abstractmethod
import os
import json
import time

class Callback:

    def __init__(self, name: str = "power_monitoring"):
        self.name = name
        self.current_phase = None

    @abstractmethod
    def __on_phase_start__(self, phase: str):
        raise NotImplementedError()

    @abstractmethod
    def __on_phase_stop__(self, phase: str):
        raise NotImplementedError()


class PowerProfilerCallback(Callback):

    def __init__(self, profiler, name: str = "power_monitoring", phases=["transfer_data_to_memory", "run_inference",
                                                                         "read_inference_results"]):
        super().__init__(name)
        self.current_phase = None
        self.profiler = profiler
        self.phases = phases

    def __on_phase_start__(self, phase: str):
        self.current_phase = phase
        if self.current_phase in self.phases:
            self.profiler.start(phase)

    def __on_phase_stop__(self, phase: str):
        if self.current_phase in self.phases:
            self.profiler.stop()
        self.current_phase = None

class TimerSyncCallback(Callback):

    def __init__(self, log_dir, name="time_sync"):
        super().__init__(name)
        self.log_dir = log_dir
        self.phases = []

    def __on_phase_start__(self, phase: str):
        self.start_time = time.time()
        self.current_phase = phase

    def __on_phase_stop__(self, phase: str):
        self.phases.append({"phase": self.current_phase, "start_time": self.start_time, "stop_time": time.time()})
        self.current_phase = None

        # save as json
        with open(os.path.join(self.log_dir, "time_sync.json"), "w") as f:
            json.dump(self.phases, f, indent=4)


