#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 15.01.2024
Description: Power Monitoring classes
"""

import time
from ppk2_api.ppk2_api import PPK2_API
import threading
import json
import os
import numpy as np


class PowerProfilerKitII:
    """
    Class that can be used to record the power consumption of a device using the Power Profiler Kit II from Nordic.
    """

    def __init__(self, port: any = "serial", name: str = "ppk2", mode="source", source_voltage: int = 5000,
                 acquisition_interval: float = 0.01, verbose: int = 0, log_dir: str = None, limit_uA=None, **kwargs):
        self.port = port
        self.name = name
        self.recordings = []
        self.current_phase = None
        self.mode = mode
        self.verbose = verbose
        self.acquisition_interval = acquisition_interval
        if log_dir is None:
            log_dir = os.path.join("logs-{}".format(time.strftime("%Y%m%d-%H%M%S")))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.limit_uA = limit_uA

        # init PPK2
        self.source_voltage = source_voltage
        self.kit = None
        self.__connect__()

    def set_source_voltage(self, int):
        """
        Set the source voltage of the Power Profiler Kit II.
        :param int: voltage in mV
        """
        self.kit.set_source_voltage(int)

    def toggle_power(self, state):
        """
        Toggle the power of the Power Profiler Kit II.
        :param state: "ON" or "OFF"
        """
        if state not in ["ON", "OFF"]:
            raise Exception(f'Unknown state {state}')
        self.kit.toggle_DUT_power(state)

    def __connect__(self):
        ppk2s_connected = PPK2_API.list_devices()
        if (len(ppk2s_connected) == 1):
            ppk2_port = ppk2s_connected[0]
            if self.verbose > 0: print(f'Found PPK2 at {ppk2_port}')
        elif len(ppk2s_connected) > 1:
            if self.verbose > 0: print("More than one PPK2 connected")
            ppk2_port = self.port
            for ppk2 in ppk2s_connected:
                if self.port == ppk2:
                    ppk2_port = ppk2
                    if self.verbose > 0: print(f'Using PPK2 at {ppk2}')
        else:
            raise Exception("No PPK2 connected")

        if self.kit:
            del self.kit
            time.sleep(3.0)
        self.kit = PPK2_API(ppk2_port)
        time.sleep(0.5)
        self.kit.get_modifiers()

        self.set_source_voltage(self.source_voltage)
        if self.mode == "ampere":
            self.kit.use_ampere_meter()
        elif self.mode == "source":
            self.kit.use_source_meter()
        else:
            raise Exception(f'Unknown mode {self.mode}')
        self.kit.toggle_DUT_power("ON")  # enable DUT power
        self.running = False
        self.buffer = []
        self.phase = None
        self.failed = False
        time.sleep(2.0)

    def __record__(self):
        """
        Concurrently record the power consumption of the device.
        """
        while self.running:
            if not self.failed:
                read_data = self.kit.get_data()
                if read_data != b'':
                    samples, raw_digital = self.kit.get_samples(read_data)
                    self.recording["samples"] += samples  # concatenate new samples
                    self.recording["digital"] += raw_digital  # concatenate new digital data
                    if self.limit_uA is not None:
                        max_val = np.asarray(samples).max()
                        if max_val > self.limit_uA:
                            print(
                                f"============= FAILED - Power ({max_val}) uA exceeds {self.limit_uA} uA=============")
                            self.failed = True
                            self.running = False  # stop recording
                    # Raw digital contains the raw digital data from the PPK2
                    # The number of raw samples is equal to the number of samples in the samples list
                    # We have to process the raw digital data to get the actual digital data

                    # TODO handle channels
                    """
                        digital_channels = self.kit.digital_channels(raw_digital)
                        for ch in digital_channels:
                            # Print last 10 values of each channel
                            print(ch[-10:])
                        print()
                    """
                time.sleep(self.acquisition_interval)

    def start(self, phase: str):
        """
        Start recording the power consumption of the device for a given phase.
        :param phase: name of the phase
        """
        # launch a thread that acquires data
        if self.running:
            raise Exception("Already recording. Stop first")
        else:
            if self.failed:
                print("Last measurement failed. Resetting the device and the PPK2")
                self.__connect__()

            self.phase = phase
            self.kit.start_measuring()  # start measuring
            time.sleep(0.1)
            self.recording = {"phase": self.phase, "samples": [], "digital": [], "start_timestamp": time.time()}
            self.running = True
            self.thread = threading.Thread(target=self.__record__, daemon=True)
            self.thread.start()

    def stop(self):
        """
        Stop recording the power consumption of the device.
        """
        self.running = False
        self.recording["stop_timestamp"] = time.time()
        self.kit.stop_measuring()  # stop measuring
        time.sleep(1.0)
        if self.thread is not None:
            self.thread.join()
            self.recordings.append(self.recording)
            with open(os.path.join(self.log_dir, f"{self.name}_recording.json"), "w") as f:
                json.dump(self.recordings, f)
            self.thread = None
            self.recording = None
        else:
            raise Exception("Not recording. Start first")
